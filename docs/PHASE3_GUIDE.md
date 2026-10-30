# Phase 3 Guide: `vla_bridge`, connecting the model to the real arm

This continues `LEARNING_GUIDE.md` (Phases 0–2) in the same style: **what** we build, **why** each choice is made, and **how** we test it.
**Status (2026-10-30): written, not yet run on ROS or hardware.** Code in `low_level_controller/open_manipulator/custom_package/`
(`vla_bridge.py`, `vla_safety.py`, `config/vla_bridge.yaml`, `launch/vla_bridge.launch.py`, `test/test_vla_safety.py`; robot-repo known issue #10).
Checked so far: 11 unit tests; an off-ROS run of the real node logic (ROS stubbed, simulated arm) against the fake server: a normal run and
all six fault drills behave as designed. That run found two bugs, both fixed: the speed cap was per *step* instead of per *second* (0.58 rad/s
seen after late steps were dropped), and a lost server connection resumed silently. Next: rung 1 on mock hardware.

---

## 1. What Phase 3 is for

So far the model runs on the server and returns action chunks, but **nothing moves the arm**. Phase 3 builds the one piece that does:
a ROS 2 node, `vla_bridge`, on the robot PC.

```
 cameras ─┐
 /joint_states ─┼─► vla_bridge ──observation──► policy server (A30, port 4000)
          │        │          ◄──(10, 5) chunk──
          │        ▼
          │   SAFETY LAYER (limits, speed cap, watchdog, e-stop, enable switch)
          │        ▼
          └── /arm_controller/joint_trajectory   (4 joints)
              /gripper_controller/gripper_cmd    (gripper, GripperCommand action)
```

**Why it's the most safety-critical part of the project:** the bridge is the **only** path by which a learned model can make
the motors move. A model can output anything: a jump, a value out of range, NaN, or nothing at all. The bridge must make all of those
harmless. The arm also has a known weak point: **joint2 overloads (`0x20`) in fast or extended moves, and the arm then collapses**.

**Why we test it with the fake policy server and not the model:** the fake server (`vla_layer/scripts/fake_policy_server.py`) sends
**known, slow, predictable** motions, and can be made to misbehave on purpose. If something goes wrong, we know it's the bridge, not
the model. No GPU is needed either: the fake server runs on the CPU, on the robot PC or the server.

**Why it lives in `low_level_controller`:** it's a ROS node that talks to `arm_controller`. That repo's rules say all our own robot
code goes in `custom_package`, and the upstream driver packages are never modified. The bridge also reuses that repo's existing facts:
`JOINT_LIMITS` and the gripper constants from `joint_commander.py`, and `forward_kinematics` from `ee_commander.py`.

---

## 2. Concepts you need

**ROS 2 node, topic, action, service.**
- A *node* is a running program.
- *Topics* are one-way message streams (e.g. `/joint_states`).
- *Actions* are requests with feedback and a result (e.g. "move the gripper to X").
- *Services* are a single request and reply (e.g. "enable the bridge: yes/no").

**`JointTrajectoryController` (JTC)**, which is what `arm_controller` is:
- It takes a **trajectory**: a list of points (joint positions), each with a `time_from_start`.
- It runs at **100 Hz** and **interpolates smoothly between points** (splines).
- When a new trajectory arrives on its topic, it **replaces** the old one from "now" onward.
- When it reaches the last point, it **holds that position**.

That last property is a free safety net: if the bridge stops sending, the arm finishes the short trajectory it has (under 0.7 s)
and then stands still.

**`GripperActionController`**, which is what `gripper_controller` is: it only accepts the `GripperCommand` **action** on
`/gripper_controller/gripper_cmd`. It has no topic. (Known issue #1 is `joint_commander` using the wrong interface for this.)

**Control rate vs. controller rate.**
- The bridge plans at **15 Hz**, the same rate as the training data: one action every 66.7 ms.
- `arm_controller` executes at **100 Hz**, filling in between our points.
- So the robot moves smoothly even though the model "thinks" much less often.

**Asynchronous inference.** A model call takes about 149 ms. If the bridge waited for each reply before moving, the arm would stutter.
Instead, a separate worker thread asks for the **next** chunk while the arm is still executing the **current** one.

---

## 3. How the node works inside

### 3.1 Threads (same pattern as `ee_commander`)
- **ROS executor thread** (`SingleThreadedExecutor` + spin thread, the repo's convention): handles incoming `/joint_states` and camera
  messages, and runs the **15 Hz control timer**.
- **Inference worker thread**: builds an observation, calls `client.infer()` over the network (it blocks for ~150 ms), and puts the
  resulting chunk in a shared buffer.
- A lock protects the shared data: the latest joint state, the latest images, and the current chunk.

**Why a separate inference thread:** a blocking network call inside the control timer would freeze the whole node, including the safety checks.

### 3.2 Building the observation (must match training exactly)

| Field sent | Source | Conversion |
|---|---|---|
| `observation/state` (5,) | `/joint_states` | **Pick joints by name**: `/joint_states` does not guarantee order. j1..j4 in rad + gripper `(0.019 − q)/0.029` clipped to [0, 1] (0 = open) |
| `observation/front_image` | main camera topic | BGR → RGB, `resize_with_pad` → 224×224 uint8, the **same function** used to build the dataset |
| `observation/wrist_image` | wrist camera topic | same |
| `prompt` | ROS parameter | Exactly the task string used in the demos |

These keys and conversions are the **contract** with `omx_pi05` on the server (D5–D8 in the plan). A cross-repo test will check that
the bridge's key names and gripper formula match `omx_pi05`, like the existing constants test.

**Before the cameras exist** (they're Phase 4), the parameter `fake_images:=true` sends black 224×224 frames. The fake server
ignores images, so Phase 3 can be fully tested now.

### 3.3 The chunk buffer and latency compensation
- Each chunk is tagged with the **time of the observation** it was computed from, `t_obs`.
- Step `k` of the chunk (k = 0…9) is meant for time `t_obs + (k+1)/15 s`.
- **Steps in the past are dropped:** when a chunk arrives ~150 ms after `t_obs`, its first ~2 steps are already in the past. Executing
  them would make the arm lag behind the model's plan.
- **When to request the next chunk:** when the remaining steps drop to `ceil(latency × 15) + 1`. At 150 ms that's 4 steps
  (~270 ms) left. The new chunk arrives with time to spare, so the arm never runs dry.
- A new chunk **replaces** the old one, and the JTC blends the switch smoothly. Joins can still look slightly jerky; a short
  cross-fade, or later real-time chunking, can fix that (`FUTURE_TODO.md`).

### 3.4 Sending commands
- **Arm:** the remaining (safe) steps of the chunk go out as **one multi-point** `trajectory_msgs/JointTrajectory` on
  `/arm_controller/joint_trajectory`, with `time_from_start = (k+1)/15 s`.
  - *Why multi-point and not one point per tick:* with several points the JTC can plan smooth velocities through all of them. Single
    points 66 ms apart make it accelerate and brake between every point, which means jerk and extra load on joint2.
- **Gripper:** the model's gripper value is continuous, between 0 and 1. A **hysteresis** turns it into commands:
  - close when it rises above **0.6**,
  - open when it falls below **0.4**,
  - only send a `GripperCommand` **when the state changes**.
  - *Why:* without hysteresis a value hovering around 0.5 would make the gripper chatter open and shut, and a goal per tick would flood
    the action server.

---

## 4. The safety layer

Every action is checked **before** it can reach a controller. These checks are written as plain Python functions in a separate module
(`vla_safety.py`), so they can be unit-tested without ROS or a robot.

| Check | Rule (starting values) | Why |
|---|---|---|
| **Starts disabled** | The node runs, but commands nothing until `/vla_bridge/enable` (`std_srvs/SetBool`) is called with `true` | Nothing moves by accident at launch. One warmup inference runs first (the ~20 s JIT compile happens before enabling). |
| **Finite values** | Any NaN or inf in a chunk → reject it, hold, disable | A broken model output must never become a motor command |
| **Joint limits** | Clamp every target to the URDF limits **minus 0.05 rad** (`JOINT_LIMITS` from `joint_commander.py`) | Never drive into the hard stops. The first observed state may already be outside the limits after hand-positioning (known issue #2), so we clamp *targets*, not the state. |
| **Speed cap** | Each point may move each joint at most `max_joint_vel × (time until that point)`, from the *measured position*. Start: **0.3 rad/s.** Capping by time matters: after late steps are dropped, the first point can be due only milliseconds from now. | The main defence against joint2 overload. Starts at the "25 %" stage of the plan; raised only after live tests. |
| **Jump check** | First step more than 0.15 rad from the current position → reject the chunk | A chunk that starts far from where the arm is signals a stale or confused prediction |
| **Workspace** | Fingertip height (forward kinematics) below `min_z` (2 cm) → cut the chunk there and hold | Don't drive the gripper into the table |
| **Watchdog** | Hold if `/joint_states` is older than 0.1 s, a camera image older than 0.2 s (when cameras are on), or no chunk has arrived for 1 s |
| **Lost connection** | Server connection drops → hold + disable. The bridge reconnects and warms up, but **only a new enable resumes** | Never resume silently on a server that may have restarted with a different model | Stale inputs mean the model is acting on an old view of the world |
| **E-stop topic** | A `true` on `/vla_bridge/estop` → hold and disable; only a new `enable` call restarts | A software e-stop anyone (or another node) can trigger. **The hardware e-stop that cuts motor power is still required** (Phase 4). |
| **Hold** | "Hold" = publish a one-point trajectory at the current measured position, then stop sending | An explicit stop, rather than relying only on the trajectory running out |
| **Current monitor** (log only at first) | Log when `/joint_states` effort (Present Current) exceeds a threshold | We already get current for free; logging first finds sensible thresholds before it's allowed to stop the arm |

**Why so many layers:** each one covers a different failure (a bad model output, a bad network, a bad sensor, a human emergency).
None of them depends on the model behaving well.

---

## 5. Shadow mode and monitoring

- **`shadow:=true`** runs the full loop (observations, inference, safety layer) but publishes to `/vla_bridge/preview` instead of the
  controllers, plus an RViz marker showing the planned fingertip path (via forward kinematics). **The arm does not move.**
  - *Why:* it lets you watch exactly what the bridge *would* do, on the real arm and with real sensors, at zero risk. It's the plan's
    first step for every new policy (fake server now, the real model in Phase 8).
- **`/vla_bridge/status`** (JSON text): enabled or disabled, last latency, steps remaining, number of holds, and the reason for the last
  hold. One glance tells you why the arm stopped.

---

## 6. Software setup on the robot PC

- The bridge runs in the **ROS 2 Jazzy Docker container** (Python 3.12). It needs only `openpi-client` (numpy, msgpack, websockets):
  ```bash
  pip install --break-system-packages -e <vla_layer>/third_party/openpi/packages/openpi-client
  ```
  This is acceptable inside a container. Alternatively, use a venv created with `--system-site-packages` so `rclpy` stays visible.
  **No JAX, no openpi, no GPU** on the robot PC.
- **Server address:** parameters `server_host` and `server_port` (4000). Either the A30's IP directly, or `localhost` through an SSH tunnel
  (`ssh -N -L 4000:localhost:4000 shradul@iras-hub`). Prefer wired Ethernet (`FUTURE_TODO.md` 1.1).
- **Launch** after the hardware launch and homing (the arm at HOME). The bridge never homes the arm itself.

---

## 7. The test ladder (each rung must pass before the next)

| Rung | What | Arm | Passes when |
|---|---|---|---|
| 0 | **Unit tests** of `vla_safety.py` (pytest, no ROS): clamp, speed cap, jump check, stale-step dropping, hysteresis, NaN rejection | none | All pass |
| 1 | **Mock hardware**: `ros2 launch custom_package open_manipulator_x_interactive.launch.py use_mock_hardware:=true` + fake server + bridge | **simulated** (ros2_control's mock hardware echoes commands back as state) | Smooth sine motion in RViz and `/joint_states`, speed ≤ 0.3 rad/s, holds when the server is killed |
| 2 | **Fault drills on mock hardware**, using the fake-server flags `--fault nan / jump / out_of_range / below_table / stall / disconnect` | simulated | Each leads to hold + disable with the right reason in `/vla_bridge/status`; `out_of_range` keeps running with joint3 clamped |
| 3 | **Shadow mode on the real arm** | real, **not moving** | The preview path in RViz is the expected small sine around HOME |
| 4 | **Live on the real arm** with the fake server at 0.3 rad/s, hand on the e-stop | real, moving slowly | Smooth motion; no joint2 overload; **kill the server → the arm stops within ~0.5 s**; e-stop topic works; gripper opens and closes on the 6 s toggle |
| 5 | **Latency from the robot PC** (`latency_probe.py --host …`), wired vs. Wi-Fi, direct vs. tunnel | none | Round trip logged in `experiments.md`; target p95 < 200 ms |

**Why a ladder:** each rung adds exactly one new risk. Mock hardware tests the logic without motors. Shadow mode tests real sensors
without motion. Live mode adds motion only once everything else has been proven.

---

## 8. What gets written

In `low_level_controller/open_manipulator/custom_package/`:
- `custom_package/vla_safety.py`: the pure safety functions (testable without ROS)
- `custom_package/vla_bridge.py`: the node (threads, observation building, chunk buffer, publishing, enable/e-stop/status, shadow mode)
- `config/vla_bridge.yaml`: parameters (server address, prompt, rates, limits, thresholds, camera topics, `fake_images`, `shadow`)
- `launch/vla_bridge.launch.py`
- `test/test_vla_safety.py`
- a `setup.py` entry point: `ros2 run custom_package vla_bridge`

In `vla_layer`:
- `--fault` flags for `scripts/fake_policy_server.py`
- a contract test that the bridge's keys and gripper formula match `omx_pi05`

Docs (that repo's rules): `openmanipulator_changes_summary.md`, `commands_guide.md`, the architecture manual (new node, topics,
service), and `known_issues.md`, marked **"not yet tested on hardware"** until rung 4 passes.

**Recommended alongside Phase 3:** fix **known issue #1** (`joint_commander`'s gripper uses the wrong interface). The bridge already uses
the correct `GripperCommand` action, but Phase 4 recording depends on the gripper working, and it's a small fix.

## 9. Phase 3 exit criteria
Rungs 0–4 pass on hardware. Rung 5 is logged. The docs are updated. Then Phase 4 (cameras, teleop, recording) starts. When the real
model is ready (Phase 8), it plugs into exactly this bridge: only `server_host` changes, from the fake server to the real one.
