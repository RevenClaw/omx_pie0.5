# OMX π0.5 Bring-Up: Progress Brief (28 Oct 2026)

**A 3.3-billion-parameter robot foundation model, running and fine-tunable on our own lab GPU in four days.**

We stood up Physical Intelligence's π0.5 vision-language-action model on the lab's NVIDIA A30 server, measured it end to end, proved
that LoRA fine-tuning fits in 24 GB, and are now wiring it to our 4-DoF OpenMANIPULATOR-X.

Shareable page: https://claude.ai/artifact/KXtasZ7hLS5E7yFiKVdBe9 (private until shared from its Share menu)

| Headline | Value |
|---|---|
| Inference round trip (p95, 100 calls) | **124 ms** (max 126 ms) |
| Peak memory, LoRA fine-tuning at batch 16 | **17.72 GiB** of the 21.24 GiB JAX limit |
| Model size | **3.3B** parameters (SigLIP 400M + Gemma 2B + 300M action expert) |
| Lines changed inside openpi | **0**. Our code wraps it, so upgrades stay easy. |

## Architecture: two machines, one clean contract

- **A30 server (live):** openpi pinned at `215abfb`, Python 3.11, JAX 0.5.3, π0.5 weights cached. No ROS, no sudo, no Docker required.
- **Robot PC (Phase 3–4):** ROS 2 Jazzy, `arm_controller` at 100 Hz, two cameras, and the planned `vla_bridge` safety node. It only needs `openpi-client`.
- **The link:** a websocket on port 4000. About 300 KB of observation goes in and a `(10, 5)` action chunk comes back.

## Phase 0 — Reproducible, guarded GPU environment ✅ (11/11 checks)

- **Pinned stack:** openpi locked to one commit; uv rebuilds PI's tested environment without root access.
- **Weights on disk:** `pi05_base` and `pi05_droid` cached on a 1.38 TB volume.
- **One-command setup:** `setup_server.sh`, `env.sh` and `.env`.
- **Shared-GPU guardrails:** `gpu_check.sh` blocks jobs when the card is busy or MIG is enabled.
- **Automated acceptance test:** `verify_env.py` runs a real bf16 matmul on the GPU and checks every import, the disk and the weights.
- **Documentation:** implementation plan (14 design decisions), experiment log, future-work list, `CLAUDE.md`.

## Phase 1 — π0.5 running live, measured, and proven trainable ✅

- **Live inference:** served the released `pi05_droid` checkpoint on port 4000.
- **Latency:** 95 ms model + ~27 ms server processing + ~1 ms network = **124 ms p95**. That's well inside the 333 ms the arm spends executing one chunk (5 steps at 15 Hz), so the next chunk is ready about 200 ms early. A one-time 18 s compile happens on the first call only.
- **Fine-tuning fits:** the real `pi05_base` weights with our LoRA setup (rank 16 / 32, 10-step chunks) peaked at **17.72 GiB at batch 16**. Batch 32 is estimated at about 23.4 GiB, so batch 16 is the setting.

## Phase 2 — Teaching openpi to speak OpenMANIPULATOR-X 🔄 (code complete, server run pending)

| Component | Status |
|---|---|
| `omx_pi05/constants.py`: every robot fact, cited to its source in `low_level_controller` | Built, tested |
| `OMXInputs` / `OMXOutputs`: cameras → model views, 5-D state, 32 → 5 action dims | Built, server test pending |
| Data and training configs: joints as deltas, gripper absolute; 200-step debug and 30k-step configs | Built, server test pending |
| Synthetic dataset generator | Built, server run pending |
| Fake policy server (safe scripted chunks for the robot bridge, no GPU) | Built, live-tested |
| Test suite | 15 tests: 11 pass locally, 4 need the openpi env |

- **Real data used:** joint limits, servo limits, HOME and exit poses, gripper range, link lengths and safety rules, all read from the controller repo. 7 automated checks fail if any of them drifts.
- **Synthetic (testing only):** fake motions, cartoon images, placeholder prompts, a 200-step test checkpoint, and a sine-wave test policy. All of it is replaced by teleop demos from Phase 4.
- **Next step:** on the A30, run tests → synthetic dataset → norm stats → 200-step training → serve → confirm `(10, 5)` action chunks.

## Ahead

| Phase | Goal |
|---|---|
| 3 | `vla_bridge` on the real arm with the fake policy server |
| 4–5 | Cameras, teleop, e-stop; 50 demos of the first task |
| 5b | ACT baseline to validate the data |
| 6–7 | Real π0.5 LoRA fine-tune (30k steps), offline replay check |
| 8–9 | Shadow mode, then live trials; 20-trial success rates |
| 10 | Multi-task language, real-time chunking, surprise monitor, planner |

*All figures come from `experiments.md` and were measured on the lab A30 on 26–27 Oct 2026, except the batch-32 estimate.*
