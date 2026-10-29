# vla_layer — π0.5 for OpenMANIPULATOR-X (GPU-server side)

Fine-tunes and serves Physical Intelligence's **π0.5** (openpi, JAX) for a 4-DoF ROBOTIS OpenMANIPULATOR-X.
This repo runs on the **GPU server (NVIDIA A30, 24 GiB, shared with other users)**. It needs **no ROS**.
The robot side (ROS 2 Jazzy, `arm_controller`, the planned `vla_bridge` node) lives in the sibling repo `../low_level_controller`
(read its `CLAUDE.md` for robot facts: joint limits, gripper units, home pose).

Read `docs/IMPLEMENTATION_PLAN.md` first. It holds the decisions (D1–D14), the phases with exit criteria, and the facts verified
against openpi. Log every training/eval run in `docs/experiments.md`. Postponed work (e.g. lowering latency) is in
`docs/FUTURE_TODO.md`. Keep all of them updated when behaviour changes.

## Layout
- `third_party/openpi/` — **git submodule, pinned commit. Never edit it** (D14). If our wrappers can't reach something, use a fork branch.
- `omx_pi05/` — our package, installed editable into openpi's venv: `constants.py` (robot facts, cited to low_level_controller;
  `tests/test_constants_match_robot.py` checks them), `gripper.py`, `kinematics.py`, `policy.py` (OMXInputs/OMXOutputs),
  `data_config.py`, `train_configs.py` (configs registered into openpi's registry; add real datasets to `DATASETS`).
- `scripts/` — `env.sh` (source first), setup/check scripts, `latency_probe.py`, `memory_test.py`, `make_synthetic_dataset.py`,
  wrappers `compute_norm_stats.py` / `train.py` / `serve.py` (call openpi's own scripts), `fake_policy_server.py` (no GPU).
- `docs/phase2_report.md` — what Phase 2 built, which robot facts are real and what is faked.
- `docs/LEARNING_GUIDE.md` — beginner-level explanation of Phases 0–2 and every design choice; keep it in sync.
- `data/`, `checkpoints/` — gitignored.

## Environment
- openpi's own uv venv at `third_party/openpi/.venv` (Python 3.11, `jax[cuda12]==0.5.3`, lerobot pinned to an old git rev that uses
  the `lerobot.common.datasets` module path). `source scripts/env.sh` activates it and sets `OPENPI_DATA_HOME` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
- `uv sync` in openpi **removes `omx_pi05`** from the venv. Re-run `scripts/setup_server.sh` (or `uv pip install -e .` from the repo root) afterwards.
- Policy server port is **4000** (`VLA_POLICY_PORT` in `env.sh`/`.env`), not openpi's default 8000, which is taken on the server.
- Run `scripts/gpu_check.sh train|serve` before any GPU job. Another user's process has held 21.6 GB before.
- Train inside `tmux`, headless.

## Conventions (fixed, see the plan's D4–D10)
- State/action 5-D `[j1, j2, j3, j4, gripper]`. Joints in rad (absolute in the dataset, trained as deltas). Gripper normalized, 0 = open, 1 = closed.
- Cameras `front` → `base_0_rgb`, `wrist` → `left_wrist_0_rgb`. RGB uint8, 224×224 via `openpi_client.image_tools.resize_with_pad`.
- 15 Hz, `action_horizon = 10`, execute 5 steps, then re-query.
- Terminal output style: emoji status prints (✅ ❌ ⚠️ 📥), matching `low_level_controller`.
- Don't commit unless asked. Commits end with the `Co-Authored-By` line.
