"""Phase 3 test server: speaks the openpi websocket protocol but returns SAFE scripted action chunks. No GPU, no model.

    python scripts/fake_policy_server.py                  # port $VLA_POLICY_PORT (4000), ~100 ms simulated latency
    python scripts/fake_policy_server.py --latency-ms 0

Needs only numpy, websockets and openpi-client (plus omx_pi05 constants), so it also runs on the robot PC.
Each chunk is (horizon, 5) in the model's output format: absolute joints j1..j4 (rad) + gripper in [0, 1] (0 = open).
Motion: slow sine waves around HOME (j1 ±0.2 rad, j3 ±0.1 rad, 20 s period), gripper toggling every 6 s.
Every step moves at most --max-vel rad/s from the *observed* state and stays JOINT_LIMIT_MARGIN inside the limits.
"""

import argparse
import asyncio
import logging
import os
import time

import numpy as np
import websockets.asyncio.server as ws_server
from openpi_client import msgpack_numpy

from omx_pi05 import constants as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fake_policy_server")


def target(t: float) -> np.ndarray:
    q = np.array(C.HOME, dtype=np.float64)
    q[0] += 0.2 * np.sin(2 * np.pi * t / 20.0)
    q[2] += 0.1 * np.sin(2 * np.pi * t / 20.0)
    gripper = float(int(t / 6.0) % 2)  # 0 open, 1 closed
    return np.append(q, gripper)


def make_chunk(state: np.ndarray, t0: float, horizon: int, fps: int, max_vel: float) -> np.ndarray:
    lo, hi = C.joint_limits_array(C.JOINT_LIMIT_MARGIN)
    q = np.clip(np.asarray(state[:4], dtype=np.float64), lo, hi)
    step = max_vel / fps
    chunk = np.empty((horizon, C.ACTION_DIM))
    for k in range(horizon):
        goal = target(t0 + (k + 1) / fps)
        q = np.clip(q + np.clip(goal[:4] - q, -step, step), lo, hi)
        chunk[k, :4], chunk[k, 4] = q, goal[4]
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLA_POLICY_PORT", 4000)))
    parser.add_argument("--horizon", type=int, default=C.ACTION_HORIZON)
    parser.add_argument("--fps", type=int, default=C.FPS)
    parser.add_argument("--max-vel", type=float, default=0.2, help="rad/s cap per joint")
    parser.add_argument("--latency-ms", type=float, default=100.0, help="simulated model time (real pi0.5 ≈ 95 ms)")
    args = parser.parse_args()

    async def handler(websocket: ws_server.ServerConnection) -> None:
        log.info(f"🔌 connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"fake_policy": True, "horizon": args.horizon, "fps": args.fps}))
        t_start = None
        async for message in websocket:
            t_recv = time.monotonic()
            obs = msgpack_numpy.unpackb(message)
            state = np.asarray(obs[C.OBS_STATE], dtype=np.float64)
            if state.shape != (C.STATE_DIM,):
                await websocket.send(f"expected {C.OBS_STATE} of shape ({C.STATE_DIM},), got {state.shape}")
                continue
            t_start = t_start or t_recv
            if args.latency_ms > 0:
                await asyncio.sleep(args.latency_ms / 1000)
            chunk = make_chunk(state, t_recv - t_start, args.horizon, args.fps, args.max_vel)
            infer_ms = (time.monotonic() - t_recv) * 1000
            await websocket.send(
                packer.pack({"actions": chunk, "policy_timing": {"infer_ms": infer_ms}, "server_timing": {"infer_ms": infer_ms}})
            )
        log.info(f"👋 connection from {websocket.remote_address} closed")

    async def run() -> None:
        async with ws_server.serve(handler, "0.0.0.0", args.port, compression=None, max_size=None) as server:
            log.info(f"✅ fake policy server on port {args.port} (horizon={args.horizon}, max_vel={args.max_vel} rad/s)")
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
