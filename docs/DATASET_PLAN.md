# Fine-tuning Dataset Plan: scripted YOLO demonstrations + leader-arm demonstrations

**Status (2026-11-05): source 1 (scripted demos) built** on the robot side and tested in simulation, not yet on the real arm:
`object_detector`, `calibrate_camera`, `scripted_demo` in `low_level_controller` (known issue #14). **Exact step-by-step procedure:
`low_level_controller/docs/finetuning_procedure.md`.** Implementation notes vs. this plan: episodes start and end at a READY pose
(gripper down above the workspace) instead of HOME, motion uses per-joint speed caps (shoulder 0.3 rad/s), the target is an
AprilTag-marked tray/bin, and the LLM parts (§3.3) are not built yet. Replaces "teleop for every demo" in `PHASE4_GUIDE.md` (Q1)
as the main data source.
Builds on what exists: `vla_bridge` + front camera (Phase 3), the recorder (`record_episodes`) and the converter
(`convert_rosbag_to_lerobot.py`, `PHASE5_GUIDE.md`). Inputs still needed from the user: §11.

Why the first stage is pick-and-place, what it is limited to, and the later stages: `FINETUNING_STAGES.md`.

**Decided (2026-11-05): we use all three data sources, in this order.**

| # | Source | Role | When | Extra robot time |
|---|---|---|---|---|
| 1 | **Template prompts + scripted YOLO demos** (§3.2 v1, §4) | The bulk: hundreds of smooth, consistent episodes → model v1 | First | Yes (the main recording effort) |
| 2 | **LLM-paraphrased prompts** (§3.2 v2) | Language variety on the *same* episodes (same motion, different wording), attached when converting | After v1 works | Little or none |
| 3 | **Leader-arm demos** (§5) | Human recoveries, ~20–30 episodes per failure type that v1 shows | After evaluating v1 on the robot | Yes, small; needs a second OM-X/replica (else a gamepad) |

Later (v3, §3.3): LLM-planned free-form tasks, once new motion primitives exist. Each round retrains on the combined data; the
`source` field in each episode's metadata shows what each part contributed.

---

## 1. The idea in one picture

Two sources of demonstrations, recorded in the same format and trained on together:

```
 A. SCRIPTED (bulk, automatic)                              B. LEADER ARM (few, human)
 task sampler ─► prompt ("pick up the red cube              human moves the leader OM-X (torque off)
                 and place it in the black bin")                    │
       │                                                            ▼
       ▼                                                     teleop node: copy joints 1:1, safety
 YOLO: where is the red cube? the black bin?                        │
       ▼                                                            │
 grasp plan (x, y, z, pitch) ─► closed-form IK ─► smooth            │
 coordinated joint trajectory (+ small random variation)            │
       │                                                            │
       └──────────────► commanded targets on /teleop/action ◄───────┘
                                   │
            front (+ wrist) camera, /joint_states, prompt ─► record_episodes ─► converter ─► π0.5 LoRA fine-tune
```

- **A (scripted)** gives **many, consistent, smooth** episodes spread over the whole table, with automatic success labels and,
  where possible, automatic resets. This is what makes the model work at all.
- **B (leader arm)** gives **a few human** episodes that contain what a script never shows: **corrections and recoveries**
  (a grasp that slips and is retried, an approach that is adjusted). These are added after the first model shows where it fails.
- Keyboard teleop is **not** used for training data: it moves one joint at a time with pauses, and the model would copy that.

---

## 2. Why this mix (for the model's performance)

| What the model needs | Scripted | Leader arm |
|---|---|---|
| Smooth, coordinated motion | ✅ | ✅ |
| Consistency (same situation → same kind of motion), so few demos are enough | ✅ best | 🟡 varies by operator and day |
| Coverage of object positions (generalizing across the table) | ✅ hundreds, unattended | 🟡 limited by human time |
| Recovery from mistakes | ❌ (partly fixed by random variation, §4.4) | ✅ natural |
| Language grounding ("red" vs "blue", "bin" vs "plate") | ✅ many object/target combinations | 🟡 few |
| Cost | the YOLO pipeline (exists) + software | a second OM-X or replica, U2D2, PSU |

The scripted expert also sets the **upper limit** for the scripted part: on these tasks the VLA learns to match it. The VLA's gain
comes from being end-to-end and closed-loop (it keeps looking while it moves), from language, and from π0.5's pretraining. The
scripted pipeline then doubles as the **baseline** the VLA is compared against (§9).

---

## 3. Tasks and prompts

### 3.1 Task structure
Every episode is one structured task, chosen by the sampler and stored in `metadata.json`:
```json
{"pick": "red cube", "place": "black bin", "prompt": "pick up the red cube and place it in the black bin"}
```
The sampler only picks objects YOLO sees right now, at reachable positions (§4.2).

### 3.2 Prompts: structure → language (not the other way)
For generating data **we choose the task**, so the prompt is produced from the structure:
- **v1: fixed templates**, one string per (pick, place) pair: `"pick up the {pick} and place it in the {place}"`.
  Decision D12 (identical prompt per task) still holds per pair. The run-time prompt must use the same template.
- **v2 (later): paraphrases**, e.g. "put the red cube in the black bin", "drop the red block into the bin", generated offline
  by an LLM from the template and reviewed once. They teach the VLA varied wording, but need more data, so not before v1 works.

### 3.3 Where an LLM fits
| Role | When | Notes |
|---|---|---|
| **Free-form command → `{pick, place}` (JSON)** for the modular pipeline | Live demos; the baseline in §9 | "pick up the red cube and place it in the black bin" → LLM → `{pick, place}` → YOLO → IK. Validate the JSON against the detected classes; refuse unknown objects. |
| **Prompt paraphrasing** | Offline, v2 | Generates wording variants for the dataset; checked by a human once |
| Choosing tasks during data generation | ❌ not used | Random sampling over detected objects is enough and fully predictable |

**Staged path to many task types (2026-11-05).** The LLM's real strength is accepting *different kinds* of instructions ("put all
the red objects in the bin", "stack the blue cube on the red one", "move the cube left of the bin"), not producing more episodes:
robot time limits the count either way. Its flexibility is bounded by the **primitive library** the scripted expert can execute
(`pick`, `place_in`, later `place_at`, `stack_on`, ...), and each task type needs ~50+ good episodes. The structured plan is the
source of truth: LLM instruction → plan → **validator** (known primitives, detected objects only) → execute → **final-scene check
against the plan's goal** (catches wrong-but-"successful" episodes, which would teach wrong word meanings) → each new
prompt → plan pair reviewed once (human or LLM back-translation).

| Stage | Prompts | Primitives | Why |
|---|---|---|---|
| v1 | templates, 1–2 task types | `pick`, `place_in` | working model fastest, no mismatch risk |
| v2 | LLM paraphrases of the templates | same | varied wording, ground truth fixed |
| v3 | LLM plans free-form instructions (validator + final-scene check) | + `place_at`, `stack_on`, multi-object | real task variety, once v1 works |

Detector note: if the YOLO model is closed-set (fixed classes such as "cube"), colours come from the box's pixels (e.g. HSV) or
from per-colour classes. An open-vocabulary detector (e.g. YOLO-World) takes the LLM's object names as text directly; consider it
if the task list grows.

### 3.4 First task set (proposal)
- Objects: 2–3 small cubes of distinct colours (graspable along the base direction, §4.3). Targets: one bin or plate.
- Tasks: `pick up the {red|blue|green} cube and place it in the {black bin}` → 2–3 prompts.
- Start with **one** task for the pilot (§7), then add colours.

---

## 4. The scripted expert (`scripted_demo` node, robot PC)

### 4.1 One episode
1. **Reset state:** arm at HOME (± small random offset, §4.4), gripper open, objects placed (§4.6).
2. **Sample a task** from the objects YOLO detects; build the prompt.
3. **Start recording** (recorder auto mode, §6), lead-in 1 s.
4. **Detect** the pick object and the target with YOLO → positions in the robot's `world` frame.
5. **Plan waypoints** in (x, y, z, pitch): above the object → down to grasp height → close → lift → above the target → lower →
   open → lift → back toward HOME.
6. **IK** each waypoint with `ee_commander.inverse_kinematics` (closed form). Check joint limits and fingertip height.
7. **Execute** as one smooth, coordinated joint trajectory (minimum-jerk between waypoints, all joints together, speed cap),
   streamed to `arm_controller`. **Publish every commanded target on `/teleop/action`** (joints rad + gripper 0..1), exactly as
   a teleop node would, so the recorder and converter need no changes. Gripper via `GripperCommand`.
8. **Verify** with YOLO after the motion: is the object inside/on the target? → `success` true/false.
9. **Stop recording** with that label; write the task structure and plan details into `metadata.json`.

### 4.2 Why not MoveIt/OMPL for the demos
OMPL is a random-sampling planner: two nearly identical situations can get very different paths, which is hard to imitate. The
closed-form IK + minimum-jerk path is deterministic, smooth and fast to compute. (MoveIt also flags the HOME pose as colliding,
known issue #12.) The scripted motion still respects the joint limits, the speed cap and a workspace box. Objects are only sampled
inside the reachable, safe region (fingertip ≥ 2 cm, not behind the base, within the radius the IK reaches with the chosen pitch).

### 4.3 Grasping with a 4-DOF arm
The OM-X can set x, y, z and pitch, but **not the gripper's rotation about the vertical axis**: yaw always follows the base
direction (`atan2(y, x)`). So the fingers always close along the line from the base to the object. Use objects that are graspable
that way (cubes placed roughly square to that line, cylinders, balls), or have the reset step place cubes aligned to it. A top-down
grasp (pitch +90°) where reachable, otherwise a tilted one.

### 4.4 Random variation (for robustness)
A script that always moves perfectly never shows the model how to get back on track. Each episode varies, within safe bounds:
start pose (small offset around HOME), approach height and grasp height (± 1–2 cm), the path between waypoints (a small random
mid-point offset), speed (± 20 %), and a short pause or none at each waypoint. Optionally, with a small probability, a deliberate
small error (approach 1–2 cm off) followed by a correction. This is the scripted substitute for human recoveries (the
"noise injection" idea in imitation learning, e.g. DART); the leader-arm demos (§5) cover real recoveries.

### 4.5 Camera-to-robot calibration
YOLO finds objects in the image; the script needs them in the robot's `world` frame. The YOLO pipeline presumably already does
this (RealSense depth + a camera pose). Check it once: place an object at a measured spot, compare YOLO's (x, y, z) with the tape
measure and with the arm's forward kinematics when the gripper touches it. Errors above ~1 cm cause missed grasps.
**The front camera must not move** after calibration; the same mount is used for recording and for running the VLA.

### 4.6 Resets
- **Automatic (preferred where possible):** after a successful episode, a *non-recorded* reset motion picks the object from the
  target and puts it at a new random reachable position (aligned per §4.3). This allows unattended runs of many episodes.
  Works best with an open target (plate, tray); picking from a deep bin may be unreliable.
- **Manual:** the operator moves the object to the next grid position the recorder suggests (as now).
- Failed episodes (object dropped elsewhere): the next detection decides where it is; if the scene is unusable, pause for a human.

---

## 5. Leader-arm demonstrations (`teleop` node, robot PC)

See the explanation in `PHASE4_GUIDE.md` T4.4. In short:
- **Hardware:** a second OM-X (or a kinematically equal replica) on its own U2D2 and power supply, torque off, read by a second
  ros2_control instance in the `/leader` namespace (read-only; upstream drivers untouched).
- **Calibration:** the leader's joint zeros match the follower's (Homing Offset procedure, `joint_offset_investigation.md`);
  checked by putting both in the same pose (≤ 0.03 rad per joint).
- **Teleop node:** copies the leader's joints 1:1 with the bridge's safety functions (limits, speed cap, fingertip height),
  an **engage step** (the follower first moves slowly to the leader), a hold-to-run switch, `/teleop/action`, and
  `/teleop/enable` (the recorder already calls it to pause and resume).
- **What to record:** not a second copy of the scripted data, but **what the first model gets wrong** (§7 stage 4): recoveries
  after a slip or a near miss, object positions where it hesitates, edge-of-workspace cases. About 20–30 episodes per failure type.
- **Without a second OM-X:** a gamepad (Cartesian x/y/z/pitch through the same IK) is the cheaper fallback for this small set.

---

## 6. Recording: changes to the recorder and metadata
- **Auto mode for `record_episodes`:** a service/topic interface (start an episode with its prompt; stop it with success
  true/false) used by `scripted_demo` instead of keyboard presses. The keyboard mode stays for leader-arm sessions.
- **One episode folder per demo, same format as now** (`PHASE5_GUIDE.md` §2): the converter needs no changes.
- **Extra metadata** (ignored by the converter, used for analysis and for choosing subsets):
  `source` (`scripted` / `leader`), `pick`, `place`, detected positions, planned waypoints, random-variation values, verification
  result, YOLO model version. Convert subsets with separate input folders, e.g. `recordings/scripted/...` and `recordings/leader/...`.

---

## 7. Stages

| Stage | What | Pass when |
|---|---|---|
| 0 | Inputs from §11; YOLO pipeline connected; camera calibration checked (§4.5); gripper verified on the real arm (known issue #1) | Calibration error ≤ 1 cm; gripper opens, closes, holds a cube |
| 1 | `scripted_demo` + recorder auto mode on mock hardware, then on the real arm at low speed | 10 consecutive scripted pick-and-places succeed; `/teleop/action` recorded |
| 2 | **Pilot:** 5–10 scripted episodes, one task → convert on the A30 → `inspect_dataset.py` (plots, contact sheet) | Images, state and action look right; gripper values bimodal |
| 3 | **Bulk:** 200–300 scripted episodes over 2–3 tasks (unattended with auto-reset where possible) → convert → norm stats → **train v1** (~32 h) | Converter skips < 10 %; training loss settles |
| 4 | **Evaluate v1** on the robot (shadow, then live at 0.3 rad/s): 20 trials per task, held-out positions included | Success rate logged in `experiments.md`; failure types listed |
| 5 | **Leader-arm demos** targeted at the failure types (20–30 each) → merge → **train v2** | v2 beats v1 on the same trials |

Sizes: ~0.5 GB of bags per 20 s episode (two cameras) → ~100–150 GB for 300 episodes on the robot PC. Convert and delete in
batches, or record the scripted bulk at 15 Hz with front + wrist only (as now) and keep `min_free_gb` checks on.

---

## 8. What gets built

| Where | Piece | Notes |
|---|---|---|
| `low_level_controller/custom_package` | `scripted_demo.py` | §4: task sampler, YOLO client, waypoints, IK, min-jerk streaming, `/teleop/action`, verification, reset |
| | `record_episodes.py` auto mode | §6: services `~/start_episode` (prompt, extra metadata), `~/stop_episode` (success) |
| | `teleop.py` + `launch/leader.launch.py` + leader ros2_control config | §5, only when a leader arm is available |
| | shared trajectory helpers (min-jerk, waypoint IK checks) in a pure module | unit-tested without ROS, like `vla_safety.py` |
| `vla_layer` | `inspect_dataset.py` | T5.3, needed for stage 2 |
| | wrist-view masking in `omx_pi05/policy.py` | ✅ built 2026-11-13: `wrist_camera=False` is the default (no wrist camera for now) |
| | paraphrase support in `convert_rosbag_to_lerobot.py` | source 2: a reviewed prompt → paraphrases file; each episode gets its template prompt or a paraphrase |
| | train config per dataset (`DATASETS`) | after stage 3 |
| YOLO pipeline (user's code) | an interface the script can call | "detect `<object>`" → position in `world` + confidence; see §11 |

---

## 9. Evaluation and the baseline
- Same trials for every system: fixed object positions (some never used in training), 20 trials per task.
- Compare: **(a)** the modular pipeline (LLM → YOLO → IK, i.e. the scripted expert driven by a prompt), **(b)** VLA v1
  (scripted data only), **(c)** VLA v2 (+ leader-arm data). Also try unseen wording and unseen positions.
- Log success rate, failure type, time to completion, and interventions in `experiments.md`.

---

## 10. Risks
| Risk | Mitigation |
|---|---|
| The VLA copies the script's blind spots (no recoveries) | Random variation (§4.4) + leader-arm recovery demos (§5) |
| Calibration error → systematic missed grasps in the data | Check §4.5 first; verification labels failed grasps; failures are excluded from training |
| Objects the 4-DOF gripper can't grasp at some orientations | Object choice and aligned resets (§4.3) |
| Too little visual variety (same table, same light) | Vary object positions, add distractor objects, keep lighting fixed but vary layout; add tasks gradually |
| Disk usage of hundreds of episodes | Convert in batches; `min_free_gb`; delete converted bags only after the dataset hash is logged |
| joint2 overload during fast scripted motions | Speed cap, elbow-first paths near HOME, current limits (`PHASE4_GUIDE.md` T4.5) |

---

## 11. Inputs needed from the user
1. **Where the YOLO pipeline code is** and how to run it (it is not in `low_level_controller` or `vla_layer`).
2. **Which camera it uses** (the RealSense front camera? depth for 3D?) and **how it gets robot coordinates** (calibration method).
3. **How it currently moves the arm** (`/ee_goal` → `ee_commander`, its own MoveIt calls, ...).
4. **Which objects/colours it detects**, and the current pick success rate.
5. **The target:** bin or plate, its size and height (for the approach height and the automatic reset).
6. **Is a second OM-X (or its servos + a U2D2) available** for the leader arm? Otherwise: gamepad for the small human set.
7. **Wrist camera:** yes (which one) or front only.
8. **LLM** for the live baseline and paraphrases: a local model or an API.
