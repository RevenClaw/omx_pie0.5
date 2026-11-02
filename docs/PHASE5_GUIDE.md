# Phase 5 Guide: from recordings to a training dataset (the converter)

Continues `PHASE4_GUIDE.md`: **what** the converter does, **why** each step is there, and **how** to run and test it.
It implements `IMPLEMENTATION_PLAN.md` T5.1 (convert), T5.2 (features) and T5.4 (dataset hash).
**Status (2026-11-02): converter written and tested** on synthetic recordings and on bags written by real `ros2 bag record`
(plain, mcap `zstd_fast` and file-zstd compression), including a write and read-back with openpi's pinned LeRobot version.
The recorder (`low_level_controller` `record_episodes`, 2026-11-02) writes this format; its mock-hardware recordings convert.
Not yet run on real teleop demos (they need the Phase 4 teleop node). Still to do: `inspect_dataset.py` (T5.3).

---

## 1. What the converter is for

The robot PC records **rosbags**: every message on every topic, each stream at its own rate (joints 100 Hz, teleop 30 Hz,
cameras 15 Hz), full-size camera frames. Training needs a **LeRobot dataset**: one row every 1/15 s with exactly what the
model sees and what it should output. The converter turns one into the other:

```
 robot PC                                   A30 server
 recorder ──► episode_0000/ (rosbag2 + metadata.json) ──copy──► convert_rosbag_to_lerobot.py ──► $HF_LEROBOT_HOME/local/omx_<task>
             episode_0001/ ...                                    read → align 15 Hz → trim → check → write      │
                                                                                                                 ▼
                                                                                   compute_norm_stats.py → train.py (Phase 6)
```

It runs on the A30 in openpi's environment. **No ROS is needed there:** the bags are read with `rosbags`, a pure-Python library.

---

## 2. What it expects (the contract with the recorder)

One folder per episode, anywhere under the input folder:
```
pick_cube/
  episode_0000/
    metadata.yaml           ← written by rosbag2
    episode_0000_0.mcap     ← the recording (mcap or sqlite3; compressed or not)
    metadata.json           ← written by the recorder: {"task": "pick up the red cube ...", "success": true, ...}
  episode_0001/ ...
```

| Stream | Topic (default) | Message | Content |
|---|---|---|---|
| state | `/joint_states` | `sensor_msgs/JointState` | `joint1..4` (rad) and `gripper_left_joint` (m), any order; converted to gripper 0 open..1 closed (D6) |
| action | `/teleop/action` | `sensor_msgs/JointState` | `joint1..4` (rad) and `gripper` (already 0..1): **the target the teleop node commanded** |
| front | `/camera/front/color/image_raw` | `sensor_msgs/Image` | `rgb8` or `bgr8`, any size (we record 640×480) |
| wrist | `/camera/wrist/image_raw` | `sensor_msgs/Image` | same; with `--no-wrist`, black frames instead |

Every message must carry a **header stamp** (the time it was measured or sent); all our sources do. Topic names can be changed
with `--state-topic`, `--action-topic`, `--front-topic`, `--wrist-topic`.

---

## 3. How it works, step by step (`omx_pi05/dataset.py`)

**1. Read.** Every message of the four topics, sorted by header stamp. Each camera frame is converted to RGB and immediately
shrunk to 224×224 with `resize_with_pad`, **the same function `vla_bridge` uses at run time** (D8). Training and the robot
therefore see identical images, and an episode stays small in memory.

**2. Align onto a 15 Hz grid.** The grid starts when all four streams have started and ends when the first one stops. For each
grid time:
```
 grid (15 Hz)     |         |         |         |
 /joint_states    ||||||||||||||||||||||||||||||||||   100 Hz → the NEAREST message (≤ 5 ms away)
 cameras          |    |    |    |    |    |    |      15 Hz  → the NEAREST frame (≤ ~33 ms away)
 /teleop/action   |  |  |  |  |  |  |  |  |  |  |      30 Hz  → the LATEST message at or before the grid time
```
- **Nearest** for what the robot *observes*: the best estimate of what the arm and the cameras looked like at that moment.
- **Latest-before** for the *action*: the command that was actually in effect at that moment. Never a command from the future,
  which would teach the model to react before something happened.

**3. Trim idle time.** Operators start recording, then begin to move, and stop recording a moment after finishing. Those still
frames at both ends would teach the model to hesitate. The converter finds the first and last frames where the **action**
changes (more than 0.01 rad or 0.05 gripper) and keeps 0.3 s (`--trim-margin`) on each side. `--no-trim` keeps everything.

**4. Check.** An episode is **skipped**, with the reason printed, if:

| Problem | Why it matters | Setting |
|---|---|---|
| A topic is missing from the bag | e.g. the wrist camera wasn't running | `--no-wrist` if there is no wrist camera |
| A gap longer than 0.2 s in any stream **inside the kept part** | dropped frames or a stalled node: the data no longer shows what happened | `--max-gap` |
| The action never changes | nothing was demonstrated | — |
| Less than 2 s left after trimming | too short to be a demo | `--min-duration` |
| No task in `metadata.json` and none given | every frame needs its prompt | `--task` |
| Marked `success=false` by the operator | failed demos teach failure | `--include-failures` to keep them |

Gaps in the trimmed idle part don't count: rosbag2 often misses a few messages right after it starts.

**5. Write.** One LeRobot episode per converted episode, in openpi's pinned LeRobot format with the same features as the
synthetic dataset (`observation.images.front`, `observation.images.wrist`, `observation.state`, `action`, plus the task string).
Then `conversion_report.json` inside the dataset folder: settings, per-episode rates, gaps, time skews, trimming, the
skipped episodes and why, and the **dataset hash** (episode names, frame counts, tasks, settings and converter commit; plan T5.4).
Log that hash in `experiments.md`: norm stats and checkpoints belong to exactly that data.

---

## 4. Commands

### 4.1 Once on the A30: get the code and `rosbags`
```bash
cd ~/vla/openpi/omx_pie0.5 && git pull
source scripts/env.sh
uv pip install "rosbags==0.11.9"          # or re-run scripts/setup_server.sh, which now installs it too
python -m pytest -q tests/test_converter.py   # expect: 10 passed
```
`uv sync` in openpi removes `rosbags` (and `omx_pi05`) again; re-run `scripts/setup_server.sh` after any `uv sync`.

### 4.2 Try it on synthetic recordings (no robot needed)
```bash
python scripts/make_synthetic_bags.py --episodes 4 --gap-episode 2 --fail-episode 3     # → data/rosbags/synthetic/, ~300 MB per episode
python scripts/convert_rosbag_to_lerobot.py data/rosbags/synthetic --dry-run           # check only, writes nothing
python scripts/convert_rosbag_to_lerobot.py data/rosbags/synthetic --repo-id local/omx_synthetic_bags
```
Expected: episodes 0 and 1 converted (about 137 frames, 9.1 s of 10.9 s each), episode 2 skipped with
`front has a 0.5x s gap`, episode 3 skipped as `marked success=false`, then a summary with the dataset hash and the task.
Delete the test data afterwards: `rm -rf data/rosbags/synthetic $HF_LEROBOT_HOME/local/omx_synthetic_bags`.

### 4.3 Real recordings
```bash
# On the robot PC: copy the episodes to the A30 (the folder the recorder writes to; rsync resumes if interrupted)
rsync -avP <recordings>/pick_cube/ shradul@iras-hub:~/vla/openpi/omx_pie0.5/data/rosbags/pick_cube/

# On the A30
python scripts/convert_rosbag_to_lerobot.py data/rosbags/pick_cube --dry-run            # read every skip reason first
python scripts/convert_rosbag_to_lerobot.py data/rosbags/pick_cube --repo-id local/omx_pick_cube
```
Useful options: `--task "<prompt>"` (override metadata), `--no-wrist`, `--include-failures`, `--overwrite` (replace the dataset),
`--max-gap 0.3`, `--no-trim`. `python scripts/convert_rosbag_to_lerobot.py --help` lists all of them.

### 4.4 Then: train on it (Phase 6)
1. Add one line to `DATASETS` in `omx_pi05/train_configs.py`: `"pick_cube": "local/omx_pick_cube",`. That creates the config
   `pi05_omx_lora_pick_cube`.
2. `python scripts/compute_norm_stats.py pi05_omx_lora_pick_cube` (again whenever the dataset changes).
3. `bash scripts/gpu_check.sh train && python scripts/train.py pi05_omx_lora_pick_cube --exp-name=pick_cube_v1` (in `tmux`).

---

## 5. Sizes
- **Bags:** about 0.5 GB per 20 s episode with two raw 640×480 cameras (28 MB/s) before compression; ~25 GB per 50 episodes.
  The synthetic episodes are ~300 MB each (11 s).
- **Dataset:** much smaller, since frames are stored as 224×224 PNG. Expect on the order of 1 GB for 50 real episodes
  (an estimate; the synthetic test images compress far better than real ones).
- Converting takes about 1 s per episode on the laptop.

---

## 6. Notes for the recorder (Phase 4, T4.6)
- Start the bag about **1 s before** the operator moves: rosbag2 can miss messages in its first fraction of a second, and the
  idle lead-in is trimmed anyway.
- Write `metadata.json` next to the bag's `metadata.yaml` (same folder), with at least `task` and `success`.
- Record `/teleop/action` as described in §2, stamped when each command is sent.
- Any of mcap (`--storage-preset-profile zstd_fast`), file compression, or no compression can be read.

## 7. Still to do in Phase 5
- `inspect_dataset.py` (T5.3): per-dimension ranges, state-vs-action plots per episode, a contact sheet of front and wrist frames,
  and a gripper histogram. Needed for the Phase 4 pilot (T4.9).
- T5.5: an export for current LeRobot (only for the optional ACT baseline, Phase 5b).
