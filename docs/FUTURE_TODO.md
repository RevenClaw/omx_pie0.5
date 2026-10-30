# Future TODO

Work that has been deliberately postponed. Each item says **when** to pick it up and **how to tell whether it worked**.
When you start an item, move it into `IMPLEMENTATION_PLAN.md` or log it in `experiments.md`. When it's done, delete it from here.

---

## 1. Lower inference latency

**Baseline (2026-10-27, `pi05_droid` on the A30, localhost:4000, see `experiments.md`):** round trip p95 = **124 ms**.
Our own fine-tuned checkpoint (2026-10-29): **149 ms** p95. Network ≈ 1 ms on localhost; nearly all of it is server-side.
The model vs. pre/post-processing split is **unknown**: openpi's `policy_timing` stops before the GPU finishes (JAX async dispatch), so only totals are reliable. The first call takes about 20 s (JIT compilation).

**Why it's postponed:** 124–149 ms is well inside the 333 ms execution window (5 steps at 15 Hz). With asynchronous chunk requests
(plan T3.3), the arm never waits for a chunk. So latency only affects how quickly the policy *reacts* to changes, not how smoothly the arm moves.

**When to pick it up:** in Phase 8 or later, **only if** the policy reacts too slowly on the real arm (for example, it's late to adjust when the
object is moved), or if the measured round trip from the robot PC goes over ~200 ms.

**Measure every change** with `scripts/latency_probe.py -n 200` (from the robot PC as well as the server) and log it in `experiments.md`.
For any change that touches the model (1.3, 1.4), also re-run the Phase 7 replay test, because faster is useless if the actions get worse.

| # | Item | Expected gain | Cost / risk | Notes |
|---|---|---|---|---|
| 1.1 | **Network:** wired Ethernet, same subnet, connect straight to port 4000 instead of through an SSH tunnel | Avoids +20–50 ms and jitter from Wi-Fi; a few ms from the tunnel | None | Do this during the robot-PC setup anyway. Keep resizing images to 224×224 on the client (≈300 KB per request instead of ≈1.8 MB). Don't JPEG-compress on a LAN. |
| 1.2 | **Profile the server time properly**: time each stage of `Policy.infer` with `jax.block_until_ready` so GPU compute is actually measured; then check why our LoRA checkpoint is ~25 ms slower than `pi05_droid` (merging the LoRA weights into the base weights for serving is one candidate) | Unknown until measured | Low; it's analysis first | Time each step inside `Policy.infer`. Check that images already arrive at 224×224 so the server resize does nothing. Only optimize what the profile shows. |
| 1.3 | **Fewer denoising steps** (`sample_kwargs={"num_steps": 5}` instead of 10 in our `serve.py`) | Maybe 30–40% of model compute (estimate; only the action-expert part shrinks) | Smoothness and precision may drop | Compare replay error at 10, 8 and 5 steps (Phase 7) before using it on the robot. |
| 1.4 | **Skip the empty third image slot.** openpi runs all 3 slots through SigLIP and masks the zero right-wrist image afterwards. | About ⅓ of the vision compute | **Breaks D14** (needs an openpi fork); must match between training and serving | Only if 1.1–1.3 aren't enough. Retrain after the change, so the model never sees the slot. |
| 1.5 | **Real-time chunking (RTC)** via LeRobot `pi05` | Smoother joins between chunks, which makes larger latency acceptable | Second training stack (Phase 10, item 2) | Hides latency better rather than reducing it |

**Not worth it for now:** TensorRT or a PyTorch + `torch.compile` port (not turnkey for π0.5, fragile, maybe 20–40% gain),
and batching or a second GPU (they raise throughput, not single-request latency).

---

## 2. Server-side policy gateway (user's proposal, 2026-10-30)

**Idea:** a second process on the server (port **4001**) sits between the robot and the model server (4000). It does the
model-side work there, and a thin ROS node on the robot PC talks to 4001 and forwards commands to the controllers.

**Assessment:**
- **Latency:** feasible, about **+2–5 ms** per chunk (one extra localhost hop re-sending ~300 KB). It must keep sending **whole chunks**;
  per-step streaming over the network would turn every network hiccup into a jerk.
- **What stays on the robot regardless:**
  - the watchdog (only the robot can notice the server or network died),
  - the speed cap and jump check (they need the *current* joint state),
  - e-stop, enable and hold,
  - the joint-limit clamp,
  - dropping late steps (needs the chunk's arrival time),
  - image resizing (keeps requests at ~300 KB).

  So the robot node stays nearly the full Phase 3 bridge.
- **What could move to the server:**
  - gripper hysteresis, chunk smoothing and real-time chunking,
  - observation validation,
  - logging and recording every inference,
  - prompt or model switching,
  - a high-level planner,
  - one shared source of key names and the gripper formula (`omx_pi05`).

**When to pick it up:** once that server-side logic is actually needed (Phase 8+: smoothing, recording, prompts, planner).
**Cheaper first option:** a policy wrapper inside our own `scripts/serve.py` on port 4000 (openpi already wraps policies, e.g. its
`PolicyRecorder`). That has no extra hop, port or process. A separate 4001 gateway is only worth it if it must outlive or switch between
model servers, e.g. hot-swapping the fake server and the real model without restarting the robot node.
**How to tell it worked:** `latency_probe.py` round trip within ~5 ms of the direct path, and the Phase 3 test ladder still passes
(especially "kill the server → the arm stops").

---

## 3. Other postponed work

Already scheduled as Phase 10 in `IMPLEMENTATION_PLAN.md`: multitask and language, LeRobot `pi05` with RTC, the surprise monitor, an external
high-level planner, and RL post-training or full fine-tuning on a rented 80 GB GPU. Add new postponed items above this section, using the same format.
