# Personal Understanding: Phase 3 (`vla_bridge`, fake server, fault drills, shadow mode) and fine-tuning

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

---

## 6. Fine-tuning

How the training data is made, from the robot to the dataset (Phase 4 → Phase 5). Details: `PHASE4_GUIDE.md`, `PHASE5_GUIDE.md`.

### 6.1 What fine-tuning needs: the core idea
π0.5 learns by **imitation**. You show it many examples of "in this situation, the right next moves were these", and it learns to
reproduce them. Each example is a pair:

```
INPUT (what the robot perceives)                    →  OUTPUT (what it should do next)
front image + wrist image + joint angles + prompt   →  the next 10 joint targets (0.67 s of motion)
```

That's exactly what `vla_bridge` sends the model at run time, plus the answer. During training the model sees the input, guesses
the 10 next moves, is corrected against what the human actually did, and adjusts its weights. Repeated over thousands of examples
(30k training steps), it learns the pattern: "when the cube looks like *this* and the arm is *here*, move *that* way."

So we must record, **at every instant, what the robot saw and what the human commanded.**

### 6.2 What exactly we record
Five things, each on its own ROS topic at its own rate:

| What | Topic | Rate | Why the model needs it |
|---|---|---|---|
| **Front camera image** | `/camera/front/color/image_raw` | 15 Hz | Sees the whole scene: where the cube and bowl are |
| **Wrist camera image** | `/camera/wrist/image_raw` (later) | 15 Hz | Close-up of gripper and object, for precise grasps |
| **Joint angles = the state** | `/joint_states` | 100 Hz | Where the arm *is* right now (4 joints in rad, gripper 0 open..1 closed) |
| **The command = the action** | `/teleop/action` | ~30 Hz | Where the human *told* the arm to go: **the answer the model learns** |
| **The task text = the prompt** | stored in `metadata.json` | once per episode | Which task this is: "pick up the red cube and place it in the bowl" |

- **Why the command and not just the joint angles?** At run time the model's output goes to the controller as a *target*, and the
  arm follows a little behind. So the model must learn to produce targets, which is exactly what the human's teleop commands are.
  Recording only where the arm ended up would teach it to always be one step late.
- **Why full 640×480 images if the model uses 224×224?** Shrinking happens later, in the converter. Keeping the originals means we
  can re-process (a different crop, or a bug fix) without re-recording.

### 6.3 What an episode is
**One complete attempt at the task, from start to finish.** For example:

```
arm at HOME, cube at grid position 3
→ operator moves arm to cube → closes gripper → lifts → moves to bowl → opens gripper → done
   (about 10–20 seconds = 150–300 moments at 15 Hz)
```

- **One episode = one folder:** the recording (a rosbag) plus `metadata.json` (task, success or failure, duration, grid position).
- **All start the same way** (HOME, gripper open), so the model learns one consistent behavior.
- **They vary the object's position** (grid 1–9), so it learns to *look* where the cube is instead of memorizing one path.
- **Failed attempts are labelled failure** and left out of training: the model copies whatever it's shown, including mistakes.
- **About 50 good episodes** of one task is the target for a first model.

### 6.4 What the recording tool does
Runs on the robot PC while a human drives the arm with teleop. Recording itself is done by ROS's standard `ros2 bag record`, which
saves every message on chosen topics into a file (a "bag"). Our recorder (`record_episodes`) wraps it to make each episode
**consistent and checked**:
- **before:** are all streams alive, is the arm at HOME, is there disk space, is the task set?
- **during:** start with a lead-in, say GO, warn immediately if a camera dies;
- **after:** a quick check of rates and gaps, a label (success / failure / discard), `metadata.json`, back to HOME.

The output is **raw**: each stream at its own rate, unaligned, full-size images. Fine for storage, not yet usable for training.

### 6.5 What the converter does
Runs on the A30 and turns raw episodes into **the training dataset**: one clean row every 1/15 s.

```
raw bag:     joints ||||||||||||||||||| (100 Hz)
             camera |    |    |    |    (15 Hz)       ──►   dataset row every 1/15 s:
             command |  |  |  |  |  |  (30 Hz)              [front 224×224, wrist 224×224, state(5), action(5), prompt]
```

For each episode it:
1. **shrinks** images to 224×224 with the same function the bridge uses (identical images in training and on the robot);
2. **aligns** all streams onto the 15 Hz grid: the nearest image and joint reading, and the command in effect at that moment;
3. **trims** the idle seconds at both ends;
4. **checks** quality and skips bad episodes with the reason;
5. **writes** the LeRobot dataset openpi trains on, plus a report and a dataset hash.

**Example of one finished row** (moment 4.2 s into episode 7):
```
front image:  224×224 picture, gripper above the cube
wrist image:  224×224 picture, cube between the fingers
state:        [0.41, -0.62, 0.95, 0.88, 0.02]   ← arm here now, gripper open
action:       [0.41, -0.60, 0.97, 0.87, 0.85]   ← human commanded: lower slightly and CLOSE
prompt:       "pick up the red cube and place it in the bowl"
```

**How training then uses it:** for each row, the model gets the inputs and must predict the actions of **that row and the next 9**
(the 10-step chunk). Behind the scenes, joints are learned as *changes* from the current pose while the gripper stays absolute
(decision D4), and every number is normalized with statistics of this dataset (norm stats).

### 6.6 What we built (2026-11-02)
**Recorder** (robot PC, `low_level_controller`):
- `record_episodes`: the terminal tool from 6.4 (pre-flight, GO, live warnings, quick check, s/f/d labels, `metadata.json`, HOME reset).
- `check_episode`: re-checks recorded sessions.
- Tested on the simulated arm with stand-in camera and teleop publishers: all labels, HOME reset, refusals, live warnings. A bug
  where rosbag2 swallowed the stop key was found and fixed. Can't record real demos yet: the teleop node (which produces
  `/teleop/action`) doesn't exist.

**Converter** (A30, `vla_layer`):
- `convert_rosbag_to_lerobot.py`: the five steps from 6.5; `make_synthetic_bags.py` makes fake episodes for testing it.
- Tested on synthetic and real-format bags, with a full write and read-back through openpi's exact LeRobot version, and on the
  recorder's own output.

**The chain today:**
```
teleop ❌  →  recorder ✅  →  converter ✅  →  training (configs exist since Phase 2) → serve → bridge ✅
```
The only missing link is teleop. Then: the 5-episode pilot, then the 50 real demos.
