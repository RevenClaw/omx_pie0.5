# Phase 4 Guide: data collection setup (cameras, teleoperation, safety, recording)

Continues `PHASE3_GUIDE.md` in the same style: **what** we build, **why**, and **how** we test it. The task list it expands is
`IMPLEMENTATION_PLAN.md` Phase 4 (T4.1–T4.6).
**Status (2026-11-02): in progress.** Done so far: the RealSense D435 front camera (`camera_front.launch.py`, 640×480 RGB, 15 Hz)
feeds `vla_bridge`, and the fake server confirmed that frames and joint angles arrive together (robot-repo known issue #11);
T4.1 gripper fix (mock-tested, real arm pending, issue #1); **T4.6 recorder and T4.8 quick check built** and tested on mock hardware
with stand-in camera/teleop publishers, its output accepted by the converter (issue #13; usage: `commands_guide.md` Step 5b).
**Decisions still needed:** §3 (teleop device, wrist camera, first task). Teleop (T4.4) is the main missing piece.

---

## 1. What Phase 4 is for

π0.5 already knows a lot about objects and language, but it has never seen *this* arm. It learns how to move the OM-X only from
**demonstrations**: recordings of a person doing the task through the robot. Phase 4 builds everything needed to record them well:

```
 person ──► LEADER device ──► teleop node ──► SAFETY (limits, speed cap) ──► follower arm (the real OM-X)
                                   │  commanded target = the ACTION
                                   ▼
 front camera ─┐
 wrist camera ─┼──────────────► RECORDER ──► one rosbag per episode + metadata (task, success)
 /joint_states ┘   (= the STATE)                  │
                                                  ▼  Phase 5 (server): convert to a LeRobot dataset → Phase 6: train
```

**Why it matters more than any other phase:** the model copies the demonstrations, including their mistakes. Hesitations, jerky
moves, a camera that shifted, different lighting, or a prompt with a typo all end up in the policy. A day spent getting the setup
right saves several days of recording data that can't be used.

Phase 4 produces **no dataset yet**: its exit is 5 good test episodes and proof that they convert. The 50 real demos are recorded
once that proof exists (end of Phase 4 / Phase 5).

---

## 2. What one demonstration contains

Every 1/15 s (15 Hz, decision D9) the dataset needs exactly what `vla_bridge` sends the model at run time, plus the answer:

| Field | Source | Format |
|---|---|---|
| `observation.images.front` | front camera | 640×480 RGB in the bag → 224×224 `resize_with_pad` in Phase 5 (D8) |
| `observation.images.wrist` | wrist camera | same |
| `observation.state` | follower `/joint_states` | `[j1..j4 rad, gripper 0 open..1 closed]` (D5, D6) |
| `action` | **the target the teleop node commanded** at that moment | same 5-D layout, absolute (trained as deltas, D4) |
| `task` | the episode's prompt | one fixed string per task (D12) |

**The action is the command, not where the arm ended up.** At run time the model's output goes to `arm_controller` as a target, and
the arm follows a little behind. So the demo must record the target the operator commanded (after the safety clamps), not the
follower's measured position. The teleop node publishes it on its own topic so the recorder can log it.

---

## 3. Decisions to make first

| # | Question | Options | Recommendation | What it changes |
|---|---|---|---|---|
| Q1 | **How does the operator drive the arm?** | (a) a **second OM-X** as a leader arm with torque off, moved by hand; the follower copies its joints 1:1. (b) a GELLO-style 3D-printed leader with small Dynamixels. (c) a **gamepad**, moving the gripper tip in x/y/z/pitch through the closed-form IK `ee_commander` already has. (d) the existing keyboard teleop (`open_manipulator_teleop`). | **(a)** if a second OM-X (or its servos and a second U2D2) is available. Otherwise **(c)**. Avoid (d): fixed-step key presses give jerky, hesitant demos. Also avoid hand-guiding the follower itself with torque off: your hand would be in every image, and there would be no separate command to record. | The teleop node (T4.4), the hardware list, demo quality, and how fast you can record |
| Q2 | **Wrist camera?** | (a) a small USB camera on the gripper (link 5). (b) front camera only. | **(a)**. It shows the gripper and object up close, which matters most for grasping, and the plan (D7) assumes it. | With (b), `omx_pi05` `OMXInputs` must mask the wrist slot (`image_mask=False`) for training and serving. That's a small `vla_layer` change, but it must be decided **before** recording, because datasets with and without the wrist view are not interchangeable. |
| Q3 | **First task and its exact prompt** | e.g. `"pick up the red cube and place it in the bowl"` | One simple pick-and-place with a single object type, inside the arm's comfortable reach (well inside the joint2 overload zone) | Workspace layout, episode protocol, and the prompt string, which must be identical in every episode and at run time |
| Q4 | **Where are recordings stored?** | robot PC disk, then copied to the A30 | Check free space first: raw frames take about 25 GB per 50 episodes (§4, T4.6) | Recorder output path, transfer script |

---

## 4. Work items

### T4.1 Gripper path (known issue #1)
- **What:** `joint_commander.py` sends the gripper a `FollowJointTrajectory` goal, but `gripper_controller` only serves the
  `GripperCommand` action (`/gripper_controller/gripper_cmd`). Fix it the same way `vla_bridge` already does.
- **Why:** teleop and recording need a working, tested gripper on hardware. The bridge's gripper path has only run on mock hardware.
- **Test:** open and close from `joint_commander` and from `vla_bridge` on the real arm. Check that `/joint_states`
  `gripper_left_joint` reaches about 0.019 m (open) and about −0.010 m (closed), or stops on the object.

### T4.2 Front camera: finish the setup
- **Mount it rigidly** where it sees the whole workspace and the gripper at every point of the task. Mark its position on the table
  (tape plus a photo). **It must never move** between demos and evaluation: the model learns the scene from this exact viewpoint.
- **USB 3:** it currently links at USB 2.1. Use a USB 3 port and cable (fine now for color at 15 Hz, needed for depth or more).
- **Fixed exposure and white balance** (T4.2 in the plan). Auto modes change brightness and colour from frame to frame and with the
  arm's position. Add to `camera_front.launch.py`: `rgb_camera.enable_auto_exposure: false`, `rgb_camera.exposure`,
  `rgb_camera.gain`, `rgb_camera.enable_auto_white_balance: false`, `rgb_camera.white_balance`. Choose the values once in RViz or
  `realsense-viewer`, under the lab's normal lighting, and keep the **same values** for recording and for running the policy.
- **Test:** 30 s of footage while the arm moves through the workspace; brightness must not pump.

### T4.3 Wrist camera (if Q2 = a)
- **Hardware:** a small UVC USB camera that does 640×480 at 15 or 30 fps. Mount it on link 5 so the fingertips are at the bottom
  of the image. Leave cable slack through the full range of joints 4 and 1, and tie the cable down so it can't snag.
- **Launch:** `camera_wrist.launch.py` with `usb_cam` (already in the Docker image): 640×480, 15 fps, RGB output, publishing
  `/camera/wrist/image_raw`. Address the camera by its stable path `/dev/v4l/by-id/...`, not `/dev/videoN`, which can change after a replug.
- **Fixed exposure and white balance** through `v4l2-ctl` controls or `usb_cam` parameters, as for the front camera.
- **Bridge:** set `wrist_image_topic: /camera/wrist/image_raw` in `vla_bridge.yaml`.
- **Test:** both cameras at about 15 Hz together (`ros2 topic hz .../metadata` for the RealSense, `image_raw` for the wrist camera);
  the fake server's `📷` line shows a real wrist image instead of `black`.

### T4.4 Teleoperation node (`teleop` in `custom_package`)
Design for Q1 = (a), a leader OM-X; the gamepad variant at the end.

1. **Leader hardware:** the second arm on its own U2D2 (e.g. `/dev/ttyUSB0`). It runs a second ros2_control instance in the
   namespace `/leader`, with only a `joint_state_broadcaster` (no command controllers), and torque off through
   `/leader/dynamixel_hardware_interface/set_dxl_torque`. Launch files and configs go in `custom_package`; the upstream drivers
   stay untouched. ROBOTIS's `open_manipulator_bringup/launch/omx_l_leader_ai.launch.py` (a namespaced leader) is a useful template.
2. **Calibration:** the leader's zero must match the follower's. Use the same Homing Offset procedure as
   `low_level_controller/docs/joint_offset_investigation.md`. **Test:** put both arms in the same pose (e.g. HOME, using a printed
   jig or careful alignment); every joint must agree to within about 0.03 rad.
3. **Engage safely:** when teleop is enabled, the follower first moves **slowly** to the leader's current pose (and refuses if it's
   more than a set distance away), and only then starts copying. Without this, enabling with the arms in different poses would make
   the follower jump.
4. **Streaming:** read `/leader/joint_states`, apply the **same safety functions as the bridge** (`vla_safety.py`: joint limits − margin,
   time-based speed cap, fingertip height ≥ min_z), and send the follower a short `JointTrajectory` at about 30 Hz. The gripper
   follows the leader's gripper through `GripperCommand`, rate-limited, without the bridge's hysteresis, so the recorded value stays continuous.
5. **Record the action:** publish each commanded target on `/teleop/action` (`sensor_msgs/JointState`: `joint1..4` rad plus
   `gripper` normalized 0..1), stamped when it was sent. This is the topic Phase 5 reads as `action`.
6. **Controls:** start disabled. Enable and disable from the terminal and with a hold-to-run switch (a foot pedal or a gamepad button)
   that disables on release. Show the status like the bridge does.

**Gamepad variant (Q1 = c):** the `joy` node, then the teleop node integrates stick velocities into a target `(x, y, z, pitch)`,
solves it with `ee_commander.inverse_kinematics`, and runs the same safety and `/teleop/action` steps. A trigger drives the gripper
and a shoulder button is the hold-to-run switch.

**Test ladder for teleop:**
1. Leader alone: `/leader/joint_states` follows hand motion, values match the follower's in the same pose.
2. Leader → follower on **mock hardware**: the simulated arm in RViz copies the leader.
3. Real follower at **0.3 rad/s**, hand on the e-stop. Then raise the speed cap only as far as joint2 tolerates without overload.

### T4.5 Safety hardware
- **Hardware e-stop:** a switch that cuts the 12 V motor supply (not just software). Within reach of the operator at every session.
- **Lower Current Limit** on the XM430s (EEPROM, set with torque off through Dynamixel Wizard or a small SDK script, using the same
  access procedure as the Homing Offset work). This limits the force in a collision and protects joint2. Pick values by testing
  that the task still works (start around 60–70 % of the factory value), and record them in `known_issues.md` and the plan.
- **Speed cap in teleop** (T4.4), the engage-safely step, and the hold-to-run switch.
- **Test:** pull the e-stop during a slow teleop move: the arm must lose power. Re-power and re-home cleanly. Confirm that the
  homing node's min-stress exit pose still works.

### T4.6 Episode recorder (`record_episodes` in `custom_package`)  ✅ built 2026-11-02 (mock-tested)
As planned below, plus: a 0.3 s stream-silence check (pre-flight and live), a 60 s auto-stop, and rosbag2 run with
`--disable-keyboard-controls` and no stdin (otherwise it steals the operator's ENTER). Config: `config/record_episodes.yaml`.
A terminal tool (same style as the other nodes) that turns recording into a few key presses:
- **Before each episode:** check that every stream is fresh (front, wrist, follower `/joint_states`, `/teleop/action`, the same
  checks as the bridge's watchdog) and that the arm is at HOME. Move it there with the homing node's 2-stage trajectory if needed.
- **Enter:** starts a rosbag2 recording (mcap, zstd compression) in `<data_dir>/<task>/episode_NNNN/` of: `/joint_states`,
  `/teleop/action`, `/camera/front/color/image_raw` + `camera_info`, `/camera/wrist/image_raw` + `camera_info`,
  `/leader/joint_states` (if a leader arm is used), `/tf_static`.
- **Enter again:** stops it, then asks **s** = success, **f** = failure (kept but excluded from training), **d** = discard
  (deleted). It writes `metadata.json`: task string, episode id, success, duration, operator, camera settings, git SHA of both repos.
- **After each episode:** prints a quick check (T4.8) and returns the arm to HOME.
- **Storage:** a raw 640×480 RGB frame is about 0.92 MB, so two cameras at 15 Hz are about 28 MB/s, about 0.5 GB per 20 s episode
  before compression, and roughly 25 GB per 50 episodes. Keep raw frames (D8 keeps them in the bags so they can be re-processed).
  Check free disk space before a session.

### T4.7 Episode protocol (write it down before recording)
- **Workspace:** tape marks for the robot base, the camera, the object positions (e.g. a 3×3 grid of start positions), and the target
  (bowl, box). Take a reference photo from the front camera.
- **Every episode:** object on a grid position, arm at HOME, start recording, do the task in one smooth motion, stop shortly
  after the object is placed. Aim for 10–20 s. Don't pause mid-task: idle time teaches the policy to hesitate (trimmed in Phase 5 anyway).
- **Vary** the object's start position across the grid and its orientation a little. **Keep fixed:** camera, lighting, table,
  target position, prompt string.
- **Failures:** if a grasp fails, stop and mark it **f**. Don't fix it mid-episode on the first dataset.
- **Counts:** 5 pilot episodes (T4.9) → check → 50 good episodes of the first task (the Phase 5 exit).

### T4.8 Quick episode check (`check_episode`)  ✅ built 2026-11-02
Run automatically by the recorder and usable on its own: duration, message count and rate per topic, the largest gap per topic
(flag anything above 0.1 s for 15 Hz streams), and that `/teleop/action` and `/joint_states` cover the whole episode. It prints one
line per topic and a ✅ or ⚠️ verdict, so a bad episode is caught right away, not on the server days later.

### T4.9 Pilot: prove the 5 test episodes convert
The converter exists (`PHASE5_GUIDE.md`); the recorder must follow its input contract (§2 and §6 there).
Copy the 5 pilot episodes to the A30 and run a first version of the Phase 5 converter on them (`convert_rosbag_to_lerobot.py`,
T5.1, pulled forward), then look at the contact sheet and the state-vs-action plots (T5.3). This catches swapped cameras, BGR
images, a wrong gripper sign, timestamp problems, or an action that lags the state, **before** 50 episodes are recorded.

### T4.10 Docs
`low_level_controller`: `commands_guide.md` (camera, teleop and recording steps), architecture manual (leader namespace, new nodes and
topics), changes summary, `known_issues.md` (each new piece "not yet tested on hardware" until it is). `vla_layer`: this guide's
status, plan Phase 4, the Q1–Q3 answers recorded as decisions in `IMPLEMENTATION_PLAN.md`, `PERSONAL_UNDERSTANDING.md` if useful.

---

## 5. Order of work and test ladder

| Step | What | Pass when |
|---|---|---|
| 0 | Answer Q1–Q4; order hardware (leader parts or gamepad, wrist camera, e-stop switch) | Decisions recorded |
| 1 | T4.1 gripper fix | Open/close works on the real arm from both nodes |
| 2 | T4.2 front camera fixed, exposure locked; T4.3 wrist camera | Both at ~15 Hz; stable brightness; bridge + fake server show both images |
| 3 | T4.5 e-stop and current limits | E-stop cuts motor power; task-range moves still work |
| 4 | T4.4 teleop: leader alone → mock follower → real follower at 0.3 rad/s | Follower copies the leader smoothly; engage step prevents jumps |
| 5 | T4.6 recorder + T4.8 check on mock hardware (cameras real, arm simulated) | Bag has every topic at the expected rate; metadata correct |
| 6 | T4.7 protocol + 5 pilot episodes on the real arm | All 5 pass the quick check |
| 7 | T4.9 pilot conversion on the A30 | Contact sheet and plots look right |

---

## 6. What gets written

In `low_level_controller/open_manipulator/custom_package/`:
- `custom_package/teleop.py` (leader or gamepad → follower, safety, `/teleop/action`), reusing `vla_safety.py`
- `custom_package/record_episodes.py` and `custom_package/check_episode.py`
- `launch/leader.launch.py` + leader ros2_control config (namespace `/leader`, read-only, torque off), if Q1 = a
- `launch/camera_wrist.launch.py`; exposure and white-balance parameters in `camera_front.launch.py`
- the `joint_commander.py` gripper fix (known issue #1)
- tests for the pure parts (action message layout, gripper mapping, episode check logic)

In `vla_layer`:
- `scripts/convert_rosbag_to_lerobot.py` (first version, for the pilot) and `scripts/inspect_dataset.py` (both really Phase 5)
- if Q2 = b: wrist mask in `omx_pi05/policy.py` and its tests

---

## 7. Phase 4 exit criteria
- Gripper, both cameras (fixed exposure), e-stop and current limits work on the real arm.
- Teleop drives the real follower smoothly at a speed that doesn't overload joint2.
- **5 pilot episodes** recorded with the recorder, all passing the quick check, and they **convert** on the A30 with plots and a
  contact sheet that look right.
- Docs updated. Then the 50 real demos are recorded and Phase 5 finishes the dataset.
