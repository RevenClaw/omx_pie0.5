# Phase 2 report: OM-X integration code (2026-10-27)

**Status:** code written. The tests that don't need openpi pass on the dev laptop (11 passed). **The openpi-dependent tests and the
synthetic dry run have not been run yet**; they need the openpi env on the A30 server (see "How to run it" below).

## What was built

| File | Purpose |
|---|---|
| `omx_pi05/constants.py` | Every robot fact and pipeline constant in one place, each one cited to its source file in `low_level_controller` |
| `omx_pi05/gripper.py` | Gripper metres ⇄ [0, 1] (0 = open, 1 = closed) |
| `omx_pi05/kinematics.py` | Forward kinematics (copy of `ee_commander.forward_kinematics`), used to keep fake poses above the table |
| `omx_pi05/policy.py` | `OMXInputs` (front → `base_0_rgb`, wrist → `left_wrist_0_rgb`, masked zero `right_wrist_0_rgb`, 5-D state, prompt) and `OMXOutputs` (32 → 5 dims) |
| `omx_pi05/data_config.py` | `LeRobotOMXDataConfig`: dataset key repack, joints 1–4 as deltas with the gripper absolute (`make_bool_mask(4, -1)`), openpi's pi0.5 model transforms |
| `omx_pi05/train_configs.py` | `pi05_omx_lora_synthetic` (30k steps) and `pi05_omx_lora_debug` (200 steps), registered into openpi's config registry. A real dataset is **one new line** in `DATASETS`. |
| `omx_pi05/openpi_scripts.py` | Loads openpi's `scripts/*.py` so our wrappers can call their `main()`. openpi itself is not edited (D14). |
| `scripts/compute_norm_stats.py`, `train.py`, `serve.py` | Thin wrappers: register our configs, then run openpi's own tool. All openpi CLI flags work. `serve.py` defaults to port 4000. |
| `scripts/make_synthetic_dataset.py` | Writes the **fake** dataset (details below) |
| `scripts/fake_policy_server.py` | openpi websocket protocol with scripted **safe** chunks, for Phase 3. No GPU, no model. Tested locally: `(10, 5)` chunks, 103 ms with simulated latency. |
| `tests/` | 22 tests: gripper mapping, constants vs. robot files, fake-server safety, transforms in the real openpi pipeline, configs, fake motions |
| `scripts/env.sh` | Now also sets `HF_LEROBOT_HOME=vla_layer/data/lerobot` (where datasets live) |

## Real information used

All of it was read from `low_level_controller`. **`tests/test_constants_match_robot.py` re-reads those files** and fails if a value drifts
(that repo duplicates values, issue #5). All 7 of these checks passed on 2026-10-27.

| Fact | Value | Source (under `open_manipulator/`) |
|---|---|---|
| Arm joints, controller | `joint1..joint4`, JTC position, 100 Hz | `open_manipulator_bringup/config/open_manipulator_x/hardware_controller_manager.yaml` |
| Gripper joint | `gripper_left_joint`, GripperActionController (action only) | same |
| Joint limits (rad) | j1 [−π, π], j2 [−1.82, 1.5], j3 [−1.5, 1.55], j4 [−1.7, 1.97] | `open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro` (j2/j3 widened in issue #2) |
| Servo position limits (ticks) | j1 0–4095, j2 862–3506, j3 712–3094, j4 715–3340. The URDF limits lie inside them. | `open_manipulator_description/ros2_control/open_manipulator_x_position.ros2_control.xacro` |
| Gripper range | open 0.019 m, close −0.010 m (URDF [−0.011, 0.020]) | `custom_package/joint_commander.py`, URDF |
| HOME | `[0.6090, -1.3622, 1.3898, -0.0154]` | `open_manipulator_bringup/config/open_manipulator_x/initial_positions.yaml` |
| Min-stress exit pose | `[0.4633, -1.8190, 1.4972, 0.4019]`. With torque off, j2 rests at −1.839, outside its limit. | `custom_package/interactive_homing_node.py`, `known_issues.md` #2 |
| Link lengths | JOINT1_X 0.012, JOINT2_Z 0.0595, upper arm (0.024, 0.128), forearm 0.124, gripper 0.126 m | `custom_package/ee_commander.py` |
| URDF max velocity | 4.8 rad/s (not safe in practice, see below) | URDF |

**Safety constraints from the docs**, recorded in `constants.py` for the Phase 3 bridge:
- **joint2 overload `0x20`:** the shoulder overloads in fast or extended moves and the arm collapses. Keep speeds low and lift the elbow first (architecture manual).
- **Speed cap:** start at **0.3 rad/s** with a **0.05 rad** margin inside the joint limits (plan T3.5).
- **`get_dxl_data`:** never call it on a running arm, because it stalls the control loop (issue #8).
- **`init_position:=false`:** launch with it for external control.
- **Hand-positioning:** Option 1 can leave joints outside the limits, so the first observed state may be out of range.
- **The gripper path is still broken (issue #1).** It must be fixed before recording demos (Phase 4).
- **Joint efforts are already available:** `/joint_states` carries **effort** (Present Current) as a state interface, which is useful for the later surprise monitor.

Also real: the π0.5 model and `pi05_base` weights, the LoRA setup validated by the memory test (batch 16 → 17.7 / 21.2 GiB), openpi's
training loop, norm stats and server, the dataset format (openpi's pinned LeRobot), and the image preprocessing
(`resize_with_pad` → 224×224).

## Fake data generated for this phase

| Item | How it is faked | Replaced in |
|---|---|---|
| **Joint motions** | Pick-and-place-like sequence: HOME → random pose → hold (close) → random pose → hold (open) → HOME. Smooth (min-jerk) moves, every joint ≤ 0.5 rad/s, poses inside the joint limits − 0.05 rad, and the fingertips 3–25 cm above the table at each pose (checked with the real FK) | Phase 5: real teleop demos |
| **State vs. action** | action = commanded pose. State = the same, one step later, plus 2 mrad noise. The gripper state lags and stops at 0.7 when it closes on the "cube". | Phase 5 |
| **Camera images** | A flat cartoon: brown table, red cube (carried while grasped), blue gripper square (larger when higher). Wrist view: cube + fingers. Drawn at 640×480, then the real preprocessing → 224×224. | Phase 4–5: real cameras |
| **Prompts** | 2 placeholder tasks ("… place it to the left / right") | Phase 4: the real task string |
| **Size** | 20 episodes, about 15 s each at 15 Hz (≈ 4–5k frames) | 50+ real demos |
| **Checkpoint** | 200-step `pi05_omx_lora_debug` run, a useless policy | Phase 6: 30k-step real run |
| **Policy for the robot** | `fake_policy_server.py`: sine waves around HOME (j1 ±0.2, j3 ±0.1 rad, 20 s period), gripper toggling every 6 s, ≤ 0.2 rad/s from the observed state | Phase 8: the real model |

**Never send the debug checkpoint's actions to the arm.** Only the fake policy server is meant for the robot in Phase 3.

## How to run it (on the server, after syncing the repo)

```bash
source scripts/env.sh
python -m pytest tests -q                                     # expect 15 passed, 7 skipped (the 7 constants checks need ../low_level_controller)
python scripts/make_synthetic_dataset.py --overwrite          # CPU, ~1–3 min
python scripts/compute_norm_stats.py pi05_omx_lora_debug      # CPU, writes assets/pi05_omx_lora_debug/local/omx_synthetic/
bash scripts/gpu_check.sh train && python scripts/train.py pi05_omx_lora_debug --exp-name=dryrun --overwrite   # GPU, ~15 min
bash scripts/gpu_check.sh serve && python scripts/serve.py policy:checkpoint \
    --policy.config=pi05_omx_lora_debug --policy.dir=checkpoints/pi05_omx_lora_debug/dryrun/199              # window 0
python scripts/latency_probe.py --env omx                     # window 1: must print actions shape=(10, 5)
```

**Phase 2 exit:** all of the above succeed, and the probe returns `(10, 5)` chunks from our own checkpoint.
