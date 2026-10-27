"""Phase 1: measure policy-server latency from wherever this runs (the server itself, or the robot PC).

    python scripts/latency_probe.py                         # DROID-shaped obs → localhost:8000
    python scripts/latency_probe.py --host 10.0.0.5 -n 200  # from the robot PC
    python scripts/latency_probe.py --env omx               # OM-X-shaped obs (Phase 2+, our own server)

Needs only `openpi-client` + numpy, so it also runs inside the ROS container.
Splits each call into model time (policy_timing), server overhead and network/serialization.
"""

import argparse
import sys
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy


def rand_image(h: int = 480, w: int = 640) -> np.ndarray:
    # Camera-sized frame, preprocessed exactly like the real client will: resize_with_pad → uint8 224×224.
    img = np.random.randint(256, size=(h, w, 3), dtype=np.uint8)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def make_obs(env: str, prompt: str) -> dict:
    if env == "droid":
        return {
            "observation/exterior_image_1_left": rand_image(),
            "observation/wrist_image_left": rand_image(),
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": prompt,
        }
    if env == "libero":
        return {
            "observation/image": rand_image(),
            "observation/wrist_image": rand_image(),
            "observation/state": np.random.rand(8),
            "prompt": prompt,
        }
    # omx: keys defined in docs/IMPLEMENTATION_PLAN.md (D5–D7)
    return {
        "observation/front_image": rand_image(),
        "observation/wrist_image": rand_image(),
        "observation/state": np.random.rand(5).astype(np.float32),
        "prompt": prompt,
    }


def pct(xs: list[float]) -> str:
    a = np.asarray(xs)
    return f"p50={np.percentile(a, 50):6.1f}  p95={np.percentile(a, 95):6.1f}  p99={np.percentile(a, 99):6.1f}  max={a.max():6.1f} ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env", choices=["droid", "libero", "omx"], default="droid")
    parser.add_argument("-n", type=int, default=100, help="timed calls")
    parser.add_argument("--warmup", type=int, default=5, help="untimed calls first (JIT compile happens on call 1)")
    parser.add_argument("--prompt", default="pick up the cube")
    args = parser.parse_args()

    print(f"🔌 Connecting to ws://{args.host}:{args.port} …")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"✅ Connected. Server metadata: {client.get_server_metadata()}")

    for i in range(args.warmup):
        t0 = time.perf_counter()
        out = client.infer(make_obs(args.env, args.prompt))
        print(f"   warmup {i + 1}/{args.warmup}: {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"📐 actions shape={out['actions'].shape} dtype={out['actions'].dtype}")

    total, model, server = [], [], []
    for _ in range(args.n):
        obs = make_obs(args.env, args.prompt)
        t0 = time.perf_counter()
        out = client.infer(obs)
        total.append((time.perf_counter() - t0) * 1000)
        model.append(out.get("policy_timing", {}).get("infer_ms", float("nan")))
        server.append(out.get("server_timing", {}).get("infer_ms", float("nan")))

    net = [t - s for t, s in zip(total, server)]
    print(f"\n⏱️  {args.n} calls, env={args.env}")
    print(f"   round trip      {pct(total)}")
    print(f"   model           {pct(model)}")
    print(f"   server total    {pct(server)}")
    print(f"   network + I/O   {pct(net)}")
    p95 = np.percentile(total, 95)
    print(f"\n{'✅' if p95 < 150 else '⚠️ '} p95 round trip {p95:.0f} ms (target < 150 ms ≈ 2 steps at 15 Hz). Log it in docs/experiments.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
