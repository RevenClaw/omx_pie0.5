# vla_layer — π0.5 for OpenMANIPULATOR-X (GPU-server side)

Fine-tunes and serves Physical Intelligence's **π0.5** (openpi, JAX) for a 4-DoF ROBOTIS OpenMANIPULATOR-X.
This repo runs on the **GPU server (NVIDIA A30, 24 GiB, shared with other users)**. It needs **no ROS**.
The robot side (ROS 2 Jazzy, `arm_controller`, the `vla_bridge` node) lives in the sibling repo `../low_level_controller`
(read its `CLAUDE.md` for robot facts: joint limits, gripper units, home pose).

Read `docs/IMPLEMENTATION_PLAN.md` first. It holds the decisions (D1–D14), the phases with exit criteria, and the facts verified
against openpi. Log every training/eval run in `docs/experiments.md`. Postponed work (e.g. lowering latency) is in
`docs/FUTURE_TODO.md`. Keep all of them updated when behaviour changes.

## Current state (2026-11-25): read this first
- **Data:** 71 recorded episodes (`guided_demo`, dataset `pick_place_2cam`, 6 cube→bin arrangements × ~10 each, front + wrist cameras,
  recorded after the gripper driver mirror, known issue #15 in the robot repo). Prompts: "pick up the {red|green|blue} cube and place it in
  the {red|green|blue} bin" (9 combinations; green→blue bin has only 3 episodes). Downloaded to the A30 with rclone to
  `data/rosbags/scripted_pick_place/pick_place_2cam/arr_*/episode_NNNN` (the converter finds episodes at any depth).
- **Converted:** `python scripts/convert_rosbag_to_lerobot.py data/rosbags/scripted_pick_place --wrist-topic /camera/wrist/color/image_raw
  --repo-id local/omx_pick_place_2cam` → 60 episodes (11 skipped: marked success=false), 22,942 frames, dataset hash `b52239a9a842`. The
  converter's default wrist topic is `/camera/wrist/image_raw` (synthetic bags), so `--wrist-topic` is required for real data.
  Needs `rosbags==0.11.9` in the venv (`uv pip install`; setup_server.sh installs it).
- **Training:** config `pi05_omx_lora_pick_place_2cam` (in `WRIST_DATASETS`), exp `pick_place_2cam_v1`, batch 16, 30k steps planned (~4 s/step),
  checkpoints every 5k in `checkpoints/pi05_omx_lora_pick_place_2cam/pick_place_2cam_v1/{5000,…,25000}`; loss ≈ 0.002 from ~17k steps (60 episodes
  = little data, overfitting is likely: judge by robot trials, not loss). Training was stopped after 25000 to test on the robot; resume with the
  same command + `--resume`, never `--overwrite`.
- **Serve:** `bash scripts/gpu_check.sh serve && python scripts/serve.py policy:checkpoint --policy.config=pi05_omx_lora_pick_place_2cam
  --policy.dir=checkpoints/pi05_omx_lora_pick_place_2cam/pick_place_2cam_v1/25000` (port 4000; the robot PC reaches it with
  `ssh -N -L 4000:localhost:4000 shradul@iras-hub`). Training and serving do not fit on the A30 together.
- **Open issue (robot repo known issue #16):** first live run: the gripper flipped close → open and the user stopped it. Formulas in
  `dataset.py` and the bridge agree (`tests/test_bridge_contract.py`), so it is probably the model's gripper output. The bridge now has
  `-p log_gripper:=true` (works in shadow mode) to print the model's raw gripper values per chunk. Next: compare checkpoints 20000/25000 and
  inspect the demo gripper action in the bags here. Robot-run results go into `docs/experiments.md` (Robot evaluations table).
- **Project report plan:** problem statement = adapting π0.5 to a low-cost 4-DoF arm with scripted (YOLO + AprilTag) demonstrations only;
  evaluation = 9 prompts × 5–10 trials, failure types logged, baseline = the scripted pipeline (60 of 71 episodes successful).

## Layout
- `third_party/openpi/` — **git submodule, pinned commit. Never edit it** (D14). If our wrappers can't reach something, use a fork branch.
- `omx_pi05/` — our package, installed editable into openpi's venv: `constants.py` (robot facts, cited to low_level_controller;
  `tests/test_constants_match_robot.py` checks them), `gripper.py`, `kinematics.py`, `policy.py` (OMXInputs/OMXOutputs),
  `data_config.py`, `train_configs.py` (configs registered into openpi's registry; add real datasets to `DATASETS`),
  `dataset.py` (rosbag2 episode → aligned 15 Hz frames; LeRobot features; needs `rosbags`, installed by setup_server.sh).
- `scripts/` — `env.sh` (source first), setup/check scripts, `latency_probe.py`, `memory_test.py`, `make_synthetic_dataset.py`,
  wrappers `compute_norm_stats.py` / `train.py` / `serve.py` (call openpi's own scripts), `fake_policy_server.py` (no GPU),
  `convert_rosbag_to_lerobot.py` (recordings → LeRobot dataset), `make_synthetic_bags.py` (fake recordings for testing it).
- `docs/phase2_report.md` — what Phase 2 built, which robot facts are real and what is faked.
- `docs/LEARNING_GUIDE.md` — beginner-level explanation of Phases 0–2 and every design choice; keep it in sync.
- `docs/PHASE3_GUIDE.md` — the same for Phase 3 (`vla_bridge` design, safety layer, test ladder). Phase 3 ✅ 2026-11-01.
- `docs/PHASE4_GUIDE.md` — Phase 4 plan: cameras, teleoperation, safety hardware, episode recorder, pilot conversion.
- `docs/FINETUNING_STAGES.md` — why v1 is pick-and-place (commands decompose into pick/place), what v1 is limited to, next stages.
- `docs/DATASET_PLAN.md` — data source plan: scripted YOLO demos (bulk) + leader-arm demos (recoveries), LLM role, stages.
- Data collection procedure (scripted YOLO demos, calibration, pipeline test, recording) lives in the robot repo:
  `../low_level_controller/docs/finetuning_procedure.md`.
- `docs/PHASE5_GUIDE.md` — the converter: input contract with the recorder, alignment/trim/checks, commands.
- `docs/PERSONAL_UNDERSTANDING.md` — the user's plain-language notes: Phase 3 (who plans vs. checks vs. moves, fake server,
  fault drills, shadow mode) and fine-tuning (what is recorded, episodes, recorder, converter).
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
  `gripper_left_joint` is **+0.019 m = open, −0.010 m = closed** (ROBOTIS's stock convention, D6). This unit's gripper is physically inverted;
  the mirror is in the robot repo's driver config (robot-repo known issue #15). `constants.GRIPPER_OPEN_M/CLOSE_M` must equal `joint_commander`'s.
- Cameras: `front` (D455, 1280×800) → `base_0_rgb`, `wrist` (D435 on the gripper, 640×480) → `left_wrist_0_rgb`. RGB uint8, 224×224 via
  `openpi_client.image_tools.resize_with_pad`. Configs of datasets in `DATASETS` mask the wrist view (`wrist_camera=False`); datasets in
  `WRIST_DATASETS` (e.g. `pick_place_2cam`) use it unmasked. Training and serving must use the same config (serving reads it).
- 15 Hz, `action_horizon = 10`, execute 5 steps, then re-query.
- Terminal output style: emoji status prints (✅ ❌ ⚠️ 📥), matching `low_level_controller`.
- Don't commit unless asked. Commits end with the `Co-Authored-By` line.
