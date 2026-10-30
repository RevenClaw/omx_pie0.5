"""Phase 3 test server: speaks the openpi websocket protocol but returns SAFE scripted action chunks. No GPU, no model.

    python scripts/fake_policy_server.py                  # port $VLA_POLICY_PORT (4000), ~150 ms simulated latency
    python scripts/fake_policy_server.py --latency-ms 0
    python scripts/fake_policy_server.py --fault nan --fault-after 10   # misbehave after 10 s (Phase 3 fault drills)

Needs only numpy, websockets and openpi-client (plus omx_pi05 constants), so it also runs on the robot PC.
Each chunk is (horizon, 5) in the model's output format: absolute joints j1..j4 (rad) + gripper in [0, 1] (0 = open).
Motion: slow sine waves around HOME (j1 ±0.2 rad, j3 ±0.1 rad, 20 s period), gripper toggling every 6 s.
Every step moves at most --max-vel rad/s from the *observed* state and stays JOINT_LIMIT_MARGIN inside the limits.

Fault drills (--fault, after --fault-after seconds), each of which vla_bridge must turn into a hold or a clamp:
  nan           a NaN in the chunk                         -> bridge: reject, hold, disable
  jump          chunk offset by +0.5 rad                   -> bridge: start-jump check, hold, disable
  out_of_range  creeps (at --max-vel) toward joint3 = 2.0   -> bridge: clamps joint3 at 1.55 - 0.05 rad (keeps running)
                (limit 1.55; fingertip stays >= 6.5 cm up), so only the clamp catches it
  below_table   heads for a pose 5.5 cm below the table    -> bridge: min_z check, hold, disable
  stall         stops answering                            -> bridge: chunk timeout, hold, disable
  disconnect    closes the connection                      -> bridge: hold, disable; reconnects, waits for re-enable
Run fault drills on MOCK HARDWARE only (use_mock_hardware:=true); below_table really drives toward the table.
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


BELOW_TABLE = np.array([0.6, 1.0, 0.8, 1.0])  # inside the joint limits, fingertip z = -0.055 m (omx_pi05 FK)
PAST_J3_LIMIT = np.array([C.HOME[0], C.HOME[1], 2.0, C.HOME[3]])  # joint3 limit is 1.55


def make_chunk(state: np.ndarray, t0: float, horizon: int, fps: int, max_vel: float, goal=None, clamp=True) -> np.ndarray:
    lo, hi = C.joint_limits_array(C.JOINT_LIMIT_MARGIN) if clamp else (np.full(4, -np.inf), np.full(4, np.inf))
    q = np.clip(np.asarray(state[:4], dtype=np.float64), lo, hi)
    step = max_vel / fps
    chunk = np.empty((horizon, C.ACTION_DIM))
    for k in range(horizon):
        g = target(t0 + (k + 1) / fps) if goal is None else np.append(goal, 0.0)
        q = np.clip(q + np.clip(g[:4] - q, -step, step), lo, hi)
        chunk[k, :4], chunk[k, 4] = q, g[4]
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLA_POLICY_PORT", 4000)))
    parser.add_argument("--horizon", type=int, default=C.ACTION_HORIZON)
    parser.add_argument("--fps", type=int, default=C.FPS)
    parser.add_argument("--max-vel", type=float, default=0.2, help="rad/s cap per joint")
    parser.add_argument("--latency-ms", type=float, default=150.0, help="simulated model time (our checkpoint ≈ 150 ms)")
    parser.add_argument("--fault", default="none",
                        choices=["none", "nan", "jump", "out_of_range", "below_table", "stall", "disconnect"])
    parser.add_argument("--fault-after", type=float, default=10.0, help="seconds of normal behaviour before the fault")
    args = parser.parse_args()

    async def handler(websocket: ws_server.ServerConnection) -> None:
        log.info(f"🔌 connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"fake_policy": True, "horizon": args.horizon, "fps": args.fps}))
        t_start, n_requests = None, 0
        async for message in websocket:
            t_recv = time.monotonic()
            obs = msgpack_numpy.unpackb(message)
            state = np.asarray(obs[C.OBS_STATE], dtype=np.float64)
            if state.shape != (C.STATE_DIM,):
                await websocket.send(f"expected {C.OBS_STATE} of shape ({C.STATE_DIM},), got {state.shape}")
                continue
            n_requests += 1
            if n_requests == 2:  # the bridge's first call is a warmup; motion and --fault-after count from enabling
                t_start = t_recv
            t_start = t_start or t_recv
            if args.latency_ms > 0:
                await asyncio.sleep(args.latency_ms / 1000)
            t = t_recv - t_start
            faulty = args.fault != "none" and t >= args.fault_after
            if faulty and args.fault == "stall":
                log.info("💤 fault=stall: no more replies")
                await asyncio.sleep(3600)
            if faulty and args.fault == "disconnect":
                log.info("✂️  fault=disconnect: closing the connection")
                await websocket.close()
                return
            goal = {"below_table": BELOW_TABLE, "out_of_range": PAST_J3_LIMIT}.get(args.fault) if faulty else None
            chunk = make_chunk(state, t, args.horizon, args.fps, args.max_vel, goal, clamp=not (faulty and args.fault == "out_of_range"))
            if faulty and args.fault == "nan":
                chunk[3, 1] = np.nan
            elif faulty and args.fault == "jump":
                chunk[:, :4] += 0.5
            if faulty:
                log.info(f"⚠️  fault={args.fault} injected")
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
