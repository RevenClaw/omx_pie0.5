# Personal Understanding: Phase 3 (`vla_bridge`, fake server, fault drills, shadow mode)

Notes from testing Phase 3 (2026-11-01), written to remember **how the pieces fit together**. Design details are in `PHASE3_GUIDE.md`.
Commands are in `low_level_controller/docs/commands_guide.md`, Step 5.

---

## 1. Who does what: planning vs. checking vs. moving

The bridge does **not** plan the trajectory. Three separate parts share the job:

```
 Policy server (π0.5 on the A30, or the fake server)      ← PLANS
     │  10 future joint positions + gripper, every ~0.67 s
     ▼
 vla_bridge (robot PC)                                    ← CHECKS + FORWARDS
     │  one JointTrajectory per plan
     ▼
 arm_controller (ros2_control, 100 Hz)                    ← FOLLOWS
     │  a position target every 10 ms
     ▼
 Dynamixel driver → servos (each runs its own position control loop)
```

1. **Policy server plans.** It looks at the camera images, the joint angles and the prompt, and returns the next 10 joint positions
   (one per 1/15 s) plus the gripper value. That is the "trajectory".
2. **`vla_bridge` checks and forwards.** It is a safety gate and translator, not a planner:
   - **Inputs:** builds the observation (joints, gripper, images, prompt) and sends it to the server.
   - **Timing:** gives each step a time, drops steps already in the past (the plan took ~150 ms to arrive), and asks for the next plan
     early enough that the current one never runs out.
   - **Safety:** rejects bad plans (NaN, start jump > 0.15 rad, fingertip below 2 cm), edits slightly-off plans (clamps joints inside
     the limits, slows steps to ≤ 0.3 rad/s), and holds the arm on any fault.
   - **Output:** the cleaned-up points as one `JointTrajectory` to `arm_controller`; a gripper command when the value crosses 0.6 / 0.4.
3. **`arm_controller` follows.** ROS 2's standard `JointTrajectoryController` (not our code) fills in between the bridge's points by
   straight-line interpolation and sends each servo a position target every 10 ms.

**In one line:** the policy decides where to go, the bridge makes it safe and well-timed, the controller and servos make it happen.

---

## 2. The fake policy server

`vla_layer/scripts/fake_policy_server.py` replaces **only the planner**. To the bridge it looks exactly like `serve.py`: same websocket
protocol, same port (4000), same reply shape `(10, 5)`. But there is no model and no GPU behind it.

- **Motion:** slow sine waves around HOME (joint 1 ±0.2 rad, joint 3 ±0.1 rad, 20 s period); the gripper toggles every 6 s.
- **Built-in safety:** each plan starts from the joint angles the bridge reports, moves ≤ 0.2 rad/s, stays 0.05 rad inside the limits.
- **Ignores** the images and the prompt for its motion. It does check them: every 5 s it logs the received joint state and the images'
  shape, type and brightness; `--save-images <dir>` saves the images as PNG plus the states in `observations.csv`.
- **Delay:** waits 150 ms per reply, like our real checkpoint (`--latency-ms` changes it).
- **Why it matters:** if the arm moves wrong with the fake server, the bug is in the bridge/controller chain, not in a model.

**Stopping `serve.py` first** (same port): Ctrl+C in its terminal/tmux window, or
`pgrep -u $USER -af scripts/serve.py` → `kill <PID>`. Check with `ss -ltn | grep 4000` (no output = free) and `nvidia-smi`.
Only kill my own processes (the GPU is shared). Alternative: leave `serve.py` running, start the fake server with `--port 4001`, and
give the bridge `-p server_port:=4001`.

---

## 3. Fault drills

Each drill makes the fake server misbehave on purpose after `--fault-after` seconds (default 10) and checks the bridge catches it.
**Only on mock hardware** (`use_mock_hardware:=true`): `below_table` really drives toward the table.

| `--fault` | What the server does | What the bridge must do |
|---|---|---|
| `nan` | A NaN in the plan | `HOLD + DISABLED: bad chunk` |
| `jump` | Whole plan shifted +0.5 rad | `HOLD + DISABLED: chunk starts 0.50 rad from the arm` |
| `out_of_range` | Pushes joint 3 toward 2.0 rad (limit 1.55) | **No fault:** joint 3 clamped at ~1.50 rad, keeps running |
| `below_table` | Heads 5.5 cm below the table | `HOLD + DISABLED: next step goes below min_z=0.02 m` |
| `stall` | Stops answering | Hold within ~1 s, reconnect, `e` works again |
| `disconnect` | Closes the connection | Hold, reconnect, warmup; moves again **only** after `e` |

How to run one: restart the fake server with `--fault <name> --fault-after 5`, restart the bridge (`q`, run again), wait for
"Warmup done", press `e`. Pass = arm stops smoothly, `s` shows `"enabled": false`, the right `last_fault`, and `holds` +1.
- Restart the bridge between drills: while disabled it doesn't notice a swapped server, so the first `e` would end in
  "policy server connection lost" (correct, but confusing).
- After `nan` / `jump` / `below_table`, a second `e` faults again at once: the server's clock already passed the fault time.

Extra stop checks: `d` key, disable service, e-stop topic (latched until `{data: false}`), Ctrl+C the fake server mid-run
(→ "connection lost"), Ctrl+C the arm launch (→ `/joint_states stale`).

**Result (2026-11-01):** fault drills tested ✅.

---

## 4. Shadow mode

Shadow mode runs the bridge exactly as live **except nothing goes to the motors**: a dry run on the robot.

| | Live (`shadow: false`) | Shadow (`shadow: true`) |
|---|---|---|
| Arm trajectories go to | `/arm_controller/joint_trajectory` → the arm moves | `/vla_bridge/preview` → nothing listens, the arm stays still |
| Gripper commands | sent | not sent |
| Safety checks, faults, status | all run | all run |
| Blue path line in RViz (`/vla_bridge/preview_path`) | shown | shown |

What shadow mode shows:
- What the bridge **would have** commanded, **after** the safety checks (late steps dropped, joints clamped, speed capped). Unsafe plans
  never appear; they become a hold and a printed fault.
- **One plan at a time**, not every instant: each plan is ~8–10 timed points covering the next ~0.6 s; the blue line is the fingertip
  path through them and is replaced when the next plan arrives. Exact numbers: `ros2 topic echo /vla_bridge/preview`.
- Because the arm never moves, every plan starts from the same pose: it shows what the bridge would command **if the robot were at
  that pose now**. The full motion only appears when the arm actually follows (live).

Why it comes before live: real joint-state timing without risk, checking a trained policy's intent later (does the line head for the
object?), and rehearsing the stop commands safely.

**Terminals:** T1 arm launch (homing option 2) → T2 fake server on the A30 → T3 RViz
(`rviz2 -d $(ros2 pkg prefix open_manipulator_description)/share/open_manipulator_description/rviz/open_manipulator.rviz`, then
Add → By topic → `/vla_bridge/preview_path` → Marker) → T4 bridge with `-p shadow:=true` (banner must say `Mode: SHADOW`).

---

## 5. Why the arm in RViz moves

RViz doesn't know whether the arm is real or simulated: it draws the robot at whatever angles `/joint_states` reports.
- **Mock hardware** has no motors; it sets each joint to the commanded position and publishes that as `/joint_states`.
- Chain: fake server plans → bridge checks and sends → `arm_controller` follows → joints move → `/joint_states` → RViz redraws.

**Observed (2026-11-01):**
- **Rung 1:** the arm in RViz moved (base ±0.2 rad, elbow ±0.1 rad, gripper toggling), so the full live pipeline works.
- **Shadow on:** the arm stayed still and only the preview path appeared.
- **Shadow off:** the arm moved. One parameter decides whether plans reach the controller; nothing else changes.

Check which mode a run used: the banner (`Mode: LIVE` / `Mode: SHADOW`) or `s` → `"shadow": true/false`.
