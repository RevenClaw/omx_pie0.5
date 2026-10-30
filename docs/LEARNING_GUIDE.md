# Learning Guide: π0.5 on the OpenMANIPULATOR-X, Phases 0–2

This guide is for someone new to the project. It explains **what** we did in each phase, **why** we did it that way, and the
concepts you need to follow along. Read it top to bottom the first time. Afterwards, use the table of contents to jump around.

Related documents: `IMPLEMENTATION_PLAN.md` (the full plan and its decisions D1–D14), `experiments.md` (every measurement),
`phase2_report.md` (Phase 2 file list), `FUTURE_TODO.md` (postponed work).

**Contents**
1. The big picture
2. Concepts you need (a short primer)
3. Phase 0: building the server environment
4. Phase 1: running the model and measuring it
5. Phase 2: teaching openpi about our robot
6. Phase 2: what you do next, step by step
7. Problems we hit and what they taught us
8. Glossary

---

## 1. The big picture

### What we are building
A robot arm that does a task from a spoken-style instruction, for example *"pick up the red cube and place it in the bowl"*,
using what it sees through two cameras.

### The pieces
- **The robot:** a ROBOTIS **OpenMANIPULATOR-X (OM-X)**. It has 4 rotating joints (`joint1`–`joint4`), a gripper, and Dynamixel
  XM430 motors. It is controlled by ROS 2 Jazzy on a **robot PC**, in the separate repo `low_level_controller`. That repo already
  moves the arm safely: homing, joint commands, Cartesian goals.
- **The brain:** **π0.5** ("pi zero point five") by Physical Intelligence. It is a **vision-language-action model (VLA)**: it takes camera
  images, a text instruction and the robot's current joint positions, and outputs the next robot motions. It has about 3.3 billion parameters.
- **The GPU server:** `iras-hub`, with an **NVIDIA A30** GPU (24 GiB of memory). π0.5 is far too big for the robot PC, so it runs here.
- **This repo (`vla_layer`):** everything on the GPU server. That covers installing π0.5, adapting it to our arm, training it and serving it.

### Why we have to train (fine-tune) it
π0.5 was pretrained by Physical Intelligence on **their** robots: two 6-DoF arms, about 18–19 numbers per action. Our arm has
**5 numbers** per action (4 joints + gripper), different cameras and a different body. No released π0.5 has ever seen an OM-X. We
confirmed this in Phase 1: the released DROID model returned actions of shape `(15, 8)`, meaning 15 steps × (7 Franka joints + gripper).
That's the wrong body for our arm.

So we must **fine-tune**: take the pretrained model, which already understands images, language and general manipulation, and
continue training it briefly on **demonstrations recorded on our own arm**. It keeps its general knowledge and learns our robot's "body".

### How the two machines talk

```
 Robot PC (ROS 2)                                   A30 GPU server (no ROS)
 ─────────────────                                  ───────────────────────
 cameras + /joint_states ──► vla_bridge ──observation (~300 KB)──►  π0.5 policy server
 arm_controller ◄── safety checks ◄──── action chunk (10 × 5) ────  (openpi, JAX, port 4000)
```

The server never runs ROS and the robot PC never runs JAX. They exchange plain arrays over a network connection. Either side can
change without breaking the other, and the two machines can use different Python versions (3.11 on the server, 3.12 in ROS Jazzy).

### The phases

| Phase | What | Status |
|---|---|---|
| 0 | Set up the server environment | ✅ Done |
| 1 | Run a released π0.5, measure speed, check that fine-tuning fits in memory | ✅ Done |
| 2 | Write the code that adapts openpi to the OM-X, and test it on fake data | 🔄 Code written and committed; server run pending |
| 3 | `vla_bridge` on the robot PC, tested with a fake policy | Next |
| 4–5 | Cameras, teleop, record and convert ~50 real demos | Next |
| 6–9 | Real fine-tune, offline checks, real-robot trials | Later |

---

## 2. Concepts you need (a short primer)

**Vision-language-action model (VLA).** A neural network that takes **images + text + robot state** and outputs **actions**.
π0.5 is built from three parts:
- **SigLIP** (~400M parameters): a vision encoder that turns each camera image into "image tokens".
- **Gemma 2B** (~2.6B parameters): a language model that reads the image tokens and the text instruction together.
- **Action expert** (~300M parameters): a smaller network that produces the actual motions.

**Action chunk.** Instead of predicting one motion at a time, π0.5 predicts a short **sequence** of future motions in one go,
called a *chunk*. Ours is `action_horizon = 10`, so one model call gives 10 future steps × 5 numbers = an array of shape `(10, 5)`.
Chunking matters for two reasons. It makes motion smoother, and it means the model doesn't need to run every control step: we
execute part of a chunk while the next one is being computed.

**Flow matching / denoising.** The action expert starts from random noise and refines it, over **10 small steps**, into a clean action
chunk, a bit like sharpening a blurry image. Those 10 steps are a large part of the ~120–150 ms each inference takes.

**State.** The robot's current situation as numbers: here `[j1, j2, j3, j4, gripper]`. Joints are in radians. The gripper is scaled to
0 = open, 1 = closed.

**Fine-tuning vs. LoRA.**
- *Full fine-tuning* updates all 3.3B parameters. It needs more than 70 GB of GPU memory, which we don't have.
- *LoRA (Low-Rank Adaptation)* freezes the big weight matrices and adds small trainable "adapter" matrices beside them, rank 16 or 32
  here. You train only the adapters (plus a few small parts). That needs far less memory, and it works well for adapting to a new robot.

**Batch size.** How many training examples the GPU processes together in one training step. Bigger batches give smoother learning but
use more memory. Our memory test chose **16**.

**JAX, XLA and JIT.** openpi is written in **JAX**, Google's numerical library. JAX **compiles** your computation into fast GPU code
the first time it runs (JIT, "just-in-time" compilation, done by **XLA**). That is why the **first** model call took 18 seconds and
every later one took about 0.1 s.

**JAX memory preallocation.** By default JAX reserves 75% of GPU memory the moment it starts, whether it needs it or not. We set
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow 90%, because LoRA training needs it. For inference and checks we set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX takes only what it actually uses.

**Checkpoint.** A saved copy of the model's weights at a point in training, for example "after 5,000 steps". `pi05_base` is the
pretrained checkpoint we start from.

**Normalization (norm stats).** Neural networks learn best when inputs and outputs fall in a similar small range, around [−1, 1].
π0.5 uses **quantile normalization**: for each number (each joint, the gripper), it finds the 1st and 99th percentile values in our
dataset and maps that range to [−1, 1]. These statistics (the "norm stats") are computed **from our own data** and saved with every
checkpoint, because the server must undo the exact same scaling on the model's output.

**Delta vs. absolute actions.**
- *Absolute:* "move joint1 to 0.65 rad."
- *Delta:* "move joint1 by +0.04 rad from where it is now."

We train the 4 joints as **deltas** and the gripper as **absolute**. The reasons are in §5.4.

**LeRobot dataset.** A standard on-disk format for robot demonstrations made by Hugging Face. It stores frames (images, state,
action, task text) grouped into episodes. openpi reads datasets in this format.

**WebSocket + msgpack.** A WebSocket is a network connection that stays open, so the client and server can send messages back and forth
without reconnecting each time. **msgpack** is a compact binary format for sending the dict of numpy arrays.

**Latency percentiles (p50, p95, p99).** Over 100 calls:
- p50 is the median.
- p95 is the time 95% of calls beat.
- p99 is the time 99% beat.

For control, **p95/p99 matter more than the average**, because the occasional slow call is what makes a robot hesitate.

---

## 3. Phase 0: building the server environment

**Goal:** a GPU server with the exact software π0.5 needs, the model weights on disk, and checks that confirm everything works before
spending hours on training. Nothing ran on the GPU yet in this phase.

### 3.1 openpi as a pinned git submodule (`third_party/openpi`, commit `215abfb`)
- **What:** openpi is Physical Intelligence's official code for π0/π0.5. We added it as a **git submodule**, which is a repo inside our
  repo that records **one exact commit**.
- **Why pinned:** our code (Phase 2) calls openpi's internal functions, and those can change between versions. Pinning means the same
  code runs every time, and we upgrade only when we choose to.
- **Why we never edit it (decision D14):** if we changed openpi's files, every upgrade would turn into a merge headache. Instead, our own
  package (`omx_pi05`) plugs into openpi from the outside.

### 3.2 The Python environment: `uv` (`scripts/setup_server.sh`)
- **What it does:**
  1. Installs `uv`, a fast Python package manager, into your home folder. No `sudo` needed.
  2. Fetches openpi at the pinned commit, including openpi's own submodules.
  3. Runs `uv sync`, which builds `third_party/openpi/.venv` from openpi's **lockfile** (`uv.lock`), a list of the exact version of every
     dependency. That gives Python 3.11 (uv downloads it), JAX 0.5.3 with its CUDA 12 libraries, Flax, and an **old pinned version of LeRobot**.
  4. Installs openpi and our `omx_pi05` into that same environment in **editable** mode: Python imports straight from the source folders,
     so code changes apply without reinstalling.
- **Why uv and not conda or Docker:** the lockfile reproduces exactly what Physical Intelligence tested, and it needs no admin rights on
  a shared server. Docker remains an option if the OS ever causes trouble (openpi is tested on Ubuntu 22.04).
- **Why `GIT_LFS_SKIP_SMUDGE=1`:** LeRobot's repo contains large test files stored with Git LFS. This flag skips downloading them.
- **Why the CUDA version didn't matter:** the JAX wheels bundle the CUDA 12 libraries themselves, so the server only needs a recent
  NVIDIA **driver**. Driver 595 (CUDA 13.2) runs CUDA 12 code fine.
- **Gotcha:** running `uv sync` again **removes** `omx_pi05`, because it isn't in openpi's lockfile. Re-run `setup_server.sh` afterwards.

### 3.3 Settings for every shell: `scripts/env.sh` + `.env`
- **What it sets:**
  - `OPENPI_DATA_HOME`: where checkpoints are cached. Yours: `/home/shradul/vla/openpi/openpi_cache`, on a disk with 1.38 TB free.
  - `HF_LEROBOT_HOME`: where datasets live (added in Phase 2; default `vla_layer/data/lerobot`).
  - `VLA_POLICY_PORT=4000`: the server port.
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
  - Finally, it **activates** the openpi venv.
- **`.env`:** machine-specific values, not committed to git, so each machine keeps its own paths.
- **Why `source` and not `bash`:** `bash script.sh` runs the script in a *child* shell, and its variables vanish when that shell exits.
  `source script.sh` runs it in **your** shell, so the variables and the activated venv stay. Scripts that only do work on disk
  (`setup_server.sh`, `gpu_check.sh`) are run with `bash`.
- **Tip:** add an alias, e.g. `alias vla='cd ~/vla/openpi/omx_pie0.5 && source scripts/env.sh'`.

### 3.4 Downloading the weights: `scripts/download_checkpoints.py`
- Uses **openpi's own downloader** (`openpi.shared.download.maybe_download`), so training and serving later find the files in exactly
  the place they expect.
- Fetches from Physical Intelligence's public Google Cloud bucket `gs://openpi-assets`. No account needed.
  - `pi05_base`: the pretrained model **we fine-tune**.
  - `pi05_droid`: a released, already fine-tuned model, used **only for testing** in Phase 1.
- Each checkpoint has `params/` (the weights) and `assets/` (that robot's normalization stats).
- Re-running is instant: already-downloaded files are skipped, and a file lock stops two processes from downloading at once.

### 3.5 Guarding the shared GPU: `scripts/gpu_check.sh`
- **Why:** the A30 is shared. On 26 Oct another user's job held 21.6 GB at 98% load. Training needs more than 22.5 GB. Starting anyway
  would crash, possibly hours into a run.
- **What it does:**
  1. Reads total and used memory and the **MIG** mode from `nvidia-smi`.
  2. Lists the processes on the GPU.
  3. Exits with an error if free memory is below **23,000 MiB** (`train`) or **9,000 MiB** (`serve`).
- **MIG** (Multi-Instance GPU) splits an A30 into smaller virtual GPUs. One slice is too small for π0.5, so the script fails if MIG is on.
- **Exit codes** let other commands depend on it: `bash scripts/gpu_check.sh train && python scripts/train.py ...` only trains if the check passes.

### 3.6 The acceptance test: `scripts/verify_env.py`
It runs 11 checks, and each one proves something specific:

| Check | Proves |
|---|---|
| Python ≥ 3.11 **from openpi's venv** | You sourced `env.sh`, not the system Python |
| GPU free (runs **first**, before JAX loads) | The card is available |
| JAX sees the GPU | Driver + CUDA libraries + JAX work together (otherwise JAX silently falls back to the CPU) |
| A real 4096×4096 **bf16** matrix multiply, result checked | The GPU computes correctly in the number format π0.5 uses |
| openpi imports, π0.5 configs exist | The pinned openpi is installed |
| `openpi-client` imports | The client used by the robot side exists |
| Old LeRobot (`lerobot.common.datasets`) imports | Our datasets must be written with **this** version |
| `omx_pi05` imports | Our package wasn't removed by `uv sync` |
| ≥ 150 GB free disk | Room for checkpoints, datasets and runs |
| Checkpoints cached | The download finished |
| ffmpeg present (warning only) | LeRobot needs it for video later |

Result: **all ✅** on 27 Oct.

---

## 4. Phase 1: running the model and measuring it

**Goal:** answer three questions before writing any robot-specific code.
1. Does π0.5 actually run on the A30?
2. Is it fast enough to control an arm?
3. Can we fine-tune it within 24 GB?

### 4.1 Serving a released model
```bash
cd third_party/openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_policy.py --env=DROID --port=4000
```
- `serve_policy.py` is openpi's server. `--env=DROID` loads the released `pi05_droid` checkpoint.
- The server loads the weights onto the GPU, opens a WebSocket on **port 4000**, and for each observation it receives, runs π0.5 and sends
  back an action chunk.
- **Why port 4000:** 8000, openpi's default, was already taken on the server. The port is now one setting (`VLA_POLICY_PORT`).
- **Why `PREALLOCATE=false`:** so `nvidia-smi` shows the **real** memory the model needs (expected about 8–10 GB) instead of a
  90% reservation.
- **Why DROID when it can't drive our arm:** this step tested the *plumbing* (GPU, server, network, timing), not the actions. The
  actions are for a Franka arm and meaningless for ours. **They were never sent to the robot.**

### 4.2 Measuring latency: `scripts/latency_probe.py`
- **How it works:**
  - It builds fake observations shaped exactly like real ones: 640×480 random frames, resized with the same `resize_with_pad` the
    real robot client will use.
  - It sends 5 **warmup** calls first, because the first call triggers JIT compilation.
  - It then times 100 calls and reports p50/p95/p99/max.
- **The breakdown:** the server reports how long the model took (`policy_timing`) and how long the whole server took (`server_timing`).
  - round trip − server time = network + serialization
  - server time − model time = server-side processing

**Results (27 Oct, on the server itself):**

| Part | Time |
|---|---|
| First call | 18.2 s (one-time JIT compile) |
| Round trip p50 / p95 / p99 | 123.6 / 124.4 / 124.6 ms (max 125.7). Extremely steady. |
| Server total (model + processing) | ~123 ms |
| Network (localhost) | ~1 ms |

**Is 124 ms fast enough? Yes, because of chunking:**
- We plan to control at **15 Hz** (one step every 66.7 ms) and execute **5 steps of each chunk** before using the next one. That's
  5 × 66.7 = **333 ms** of motion per chunk.
- If we request the next chunk while the current one is still running, it arrives after about 124 ms, roughly 200 ms before it's needed.
  **The arm never waits.**
- Latency only affects how *quickly the policy reacts* to changes, such as someone moving the cube.
- Ways to lower it further are documented in `FUTURE_TODO.md`:
  - wired Ethernet
  - profiling where the server time goes
  - fewer denoising steps
  - skipping the unused third camera slot
  - real-time chunking

  They're postponed because they aren't needed yet.

### 4.3 How the network communication works
- One WebSocket connection stays open. The server first sends metadata, then every `infer()` is one request and one reply.
- Request: two 224×224×3 images (~150 KB each), 5 state numbers and the prompt, so **~300 KB**. Reply: the action chunk, under 1 KB.
- Images are resized to 224×224 **on the robot PC**, so 300 KB travels instead of ~1.8 MB of full frames.

### 4.4 Can we fine-tune here? `scripts/memory_test.py`
- **The question:** LoRA fine-tuning needs more than 22.5 GB by openpi's README, and our card has 24 GiB. Does it fit, and at which batch size?
- **The trick:** openpi has a built-in **FakeDataset** that generates random tensors with exactly the model's input shapes. So we could
  run the **real** training loop with the **real** `pi05_base` weights and our exact LoRA model **before any dataset existed**. The
  script turns off checkpoint saving (nothing useful to save) and reports the peak memory JAX used.
- **Our model settings:**
  - `pi05=True`
  - `action_horizon=10`
  - `paligemma_variant="gemma_2b_lora"`: LoRA rank 16 on the 2B language model
  - `action_expert_variant="gemma_300m_lora"`: LoRA rank 32 on the action expert
  - `ema_decay=None`: EMA keeps a second smoothed copy of all weights, and there's no room for that on 24 GB

**What sits in GPU memory during LoRA training** (estimates; the test measures the total):

| Part | Why it's there | Approx. |
|---|---|---|
| Gemma 2B + action expert, **frozen**, stored in bf16 (2 bytes per number) | Needed for the forward pass, not trained | ~5.5 GiB |
| SigLIP vision encoder, **trained** (openpi's LoRA setup freezes only the Gemma models) | fp32 weights + gradients + 2 AdamW optimizer values = 16 bytes per parameter | ~6 GiB |
| LoRA adapters + small action projections, trained | Same 16 B/param, but small | < 1 GiB |
| Activations (grow with batch size) | Intermediate results kept for backpropagation; reduced by gradient checkpointing | the rest |

**Result:** batch 16 **fits**. Peak **17.72 GiB** of the 21.24 GiB JAX limit (83%), leaving about 3.5 GiB of margin.
- **Batch 32** would need roughly 23.4 GiB by estimate, so we didn't run it.
- **Batch 24** might just fit, but with about 0.6 GiB of margin a 30-hour run could crash late.
- **Why 16 is plenty:** with about 50 demos (~11k frames), 30k steps at batch 16 goes through the data about 40 times.

**Still pending:** the seconds per step (s/it) from that run's progress bar. It's needed to estimate real training time.

---

## 5. Phase 2: teaching openpi about our robot

**Goal:**
- Write the code that tells openpi what an OM-X is: its cameras, its 5 numbers, its gripper convention, its limits.
- Run the **entire** pipeline (dataset → norm stats → training → saved checkpoint → server → client) on **fake data** before recording
  any real demos.

**Why test on fake data first:** the pipeline has many pieces that must agree on every key name, shape, unit and library version. If
one disagrees, it fails. Better to find that out today, in minutes, than after building cameras and teleop and recording 50 demos.
Once the pipeline works on fake data, any later problem must be in the real **data**, which halves the search.

### 5.1 `omx_pi05/constants.py`: one source of truth for robot facts
- Every robot fact lives here once, each **cited to the file it came from** in `low_level_controller`:

| Fact | Value | Source |
|---|---|---|
| Joints | `joint1..joint4`, controller at 100 Hz | `hardware_controller_manager.yaml` |
| Joint limits (rad) | j1 [−π, π], j2 [−1.82, 1.5], j3 [−1.5, 1.55], j4 [−1.7, 1.97] | URDF xacro (j2/j3 widened in known issue #2) |
| Servo EEPROM limits | inside these, so the URDF limits are the binding ones | ros2_control xacro |
| Gripper | open 0.019 m, closed −0.010 m | `joint_commander.py` |
| HOME | `[0.6090, −1.3622, 1.3898, −0.0154]` | `initial_positions.yaml` |
| Exit pose | `[0.4633, −1.8190, 1.4972, 0.4019]` | `interactive_homing_node.py` |
| Link lengths | used for forward kinematics | `ee_commander.py` |

- **Pipeline decisions:**
  - `FPS = 15`
  - `ACTION_HORIZON = 10`
  - `EXECUTE_STEPS = 5`
  - `IMAGE_SIZE = 224`
  - the key names for observations and dataset features
- **Safety constraints from the docs,** for the Phase 3 bridge:
  - joint2 overloads (`0x20`) in fast or extended moves, so go slow and lift the elbow first
  - start at **0.3 rad/s**
  - stay **0.05 rad** inside the limits
  - never call `get_dxl_data` on a running arm (issue #8)
  - launch with `init_position:=false`
- **Why a separate copy instead of reading the YAML at runtime:** the server doesn't have `low_level_controller`. To keep the copy
  honest, **`tests/test_constants_match_robot.py` re-reads the controller's files and fails if any value differs**. This matters because
  that repo already duplicates values in several places (its issue #5). All 7 checks pass.
- It's pure Python + numpy, with no openpi or ROS imports, so the robot PC can use it too.

### 5.2 `omx_pi05/gripper.py`: the gripper convention
- The gripper moves in **metres**: `gripper_left_joint` runs from 0.019 (open) to −0.010 (closed).
- The model sees **[0, 1]**, with **0 = open and 1 = closed**: `norm = (0.019 − q) / 0.029`, clipped to [0, 1].
- **Why normalize:** keeps the gripper on the same scale as everything else, and matches the convention of most π datasets.
- **Why this direction matters:** an inverted gripper is one of the most common reasons a trained policy "never grasps". Tests check
  both endpoints, clipping, the round trip, and that opening gives a smaller number.

### 5.3 `omx_pi05/policy.py`: the adapter between our robot and the model
π0.5 always expects **3 image slots**: `base_0_rgb` (third-person), `left_wrist_0_rgb`, `right_wrist_0_rgb`. That's because PI's robots
have one main camera and two wrist cameras.

- **`OMXInputs`** (runs in training and when serving):
  - front camera → `base_0_rgb`
  - wrist camera → `left_wrist_0_rgb`
  - `right_wrist_0_rgb` → **black image with mask = False**, so the model ignores it (we have only one wrist)
  - passes through the 5-number state, the prompt, and the actions (training only)
  - converts LeRobot's image format (float, channels-first) into uint8 height × width × channels, which is what the client sends
- **`OMXOutputs`** (serving only): the model always outputs **32** numbers per step, because π0.5 pads every robot to 32. We keep the first 5.
- Modelled on openpi's own `libero_policy.py`, so it follows the pattern openpi expects.

### 5.4 `omx_pi05/data_config.py`: how openpi reads our dataset
openpi processes data in stages, with different stages for training and serving:

```
TRAINING:  dataset frame
           → repack (rename dataset keys to the client's keys)
           → OMXInputs → DeltaActions
           → Normalize (norm stats)
           → model transforms (resize, tokenize prompt + state, pad to 32)
           → model

SERVING:   client observation → OMXInputs → Normalize → model transforms → model
           → Unnormalize → AbsoluteActions → OMXOutputs → (10, 5) chunk to the robot
```

- **Repack:** maps our dataset feature names (`observation.images.front`, `observation.state`, `action`, …) to the names the robot
  client sends (`observation/front_image`, …). Training data then looks exactly like what the robot will send at run time.
- **Delta joints, absolute gripper** (`make_bool_mask(4, -1)` = `(True, True, True, True, False)`):
  - In training, `DeltaActions` turns each future joint target into "target − current position".
  - In serving, `AbsoluteActions` adds the current position back before anything goes to the robot.
  - *Why deltas for joints:* the model learns "how to move from here" rather than memorizing absolute angles. The numbers are
    small and centred near zero, which is easier to learn and generalizes better to different starting poses. openpi's ALOHA examples do the same.
  - *Why the gripper stays absolute:* open/closed is a command, not a small adjustment. "Close the gripper" should mean the same thing
    wherever the gripper currently is.
- **Prompt from the dataset task** (`prompt_from_task=True`): each recorded episode stores its task text, and that becomes the prompt.
- **Action chunks from the dataset** (`action_sequence_keys=("action",)`): for each frame, the loader grabs the next 10 actions
  (using `fps` to find them), which form the training target chunk.
- **Quantile normalization** is chosen automatically for π0.5. The norm stats are computed **after** the delta step, so they describe
  the delta joints.

### 5.5 `omx_pi05/train_configs.py`: the training recipes
- **`pi05_omx_lora_debug`:** 200 steps, checkpoints at steps 100 and 199. Used for the dry run on the synthetic dataset.
- **`pi05_omx_lora_synthetic`:** a full 30,000-step run, a checkpoint every 5,000 steps. It's the template for real datasets.

| Setting | Value | Why |
|---|---|---|
| Model | π0.5, LoRA rank 16 (Gemma) / 32 (action expert), horizon 10 | Exactly what the memory test proved fits |
| Starting weights | `gs://openpi-assets/checkpoints/pi05_base/params` | The pretrained base model meant for fine-tuning |
| Freeze filter | `model.get_freeze_filter()` | Freezes the Gemma models except their LoRA adapters |
| `ema_decay` | `None` | No memory for an EMA copy |
| `batch_size` | 16 | Memory test result |
| Steps / save interval | 30,000 / 5,000 | openpi's LoRA recipe. We keep every 5k checkpoint and pick the best one **on the robot**, not by loss (flow-matching loss correlates poorly with success). |
| Learning rate | openpi default: cosine schedule, 1,000 warmup steps, peak 2.5e-5 | Same as openpi's LoRA examples. (In the 200-step debug run the LR barely warms up, which is fine because that run only tests the plumbing.) |
| Paths | `vla_layer/assets/…` (norm stats) and `vla_layer/checkpoints/…` | Absolute paths, so results don't depend on which folder you run from |
| wandb | Off | Turn it on with `--wandb-enabled` after `wandb login`; otherwise the run would stop and wait for a login |

- **Adding a real dataset is one line:** add `"pick_cube": "local/omx_pick_cube"` to `DATASETS`, which creates `pi05_omx_lora_pick_cube`.
- **The registry trick:** openpi's tools look up configs by name in an internal dictionary. `register()` adds our configs there, so
  openpi's **own** command-line tools work with our names, with no fork. It's the one place we touch an openpi internal (`_CONFIGS_DICT`).
  If a future openpi version changes it, `tests/test_train_configs.py` will fail and tell us.

### 5.6 The wrapper scripts: `compute_norm_stats.py`, `train.py`, `serve.py`
- Each one is a few lines. It imports `omx_pi05.train_configs` (which registers our configs), then calls **openpi's own** script.
  `omx_pi05/openpi_scripts.py` loads openpi's `scripts/*.py` files, which aren't importable modules.
- **Why wrappers instead of copying openpi's code:** we automatically get openpi's exact behaviour, bug fixes and every command-line
  flag (`--exp-name`, `--resume`, `--batch-size`, …), and we maintain almost no code.
- `serve.py` adds `--port=$VLA_POLICY_PORT` (4000) if you don't pass a port.

### 5.7 `scripts/make_synthetic_dataset.py`: the fake dataset

| Part | Real or fake | Detail |
|---|---|---|
| File format | **Real** | Same feature names, shapes, dtypes and 15 fps, written with **openpi's pinned LeRobot**, exactly as the real converter will |
| Joint motions | Fake, but physically sensible | HOME → random pose → hold while "closing" → random pose → hold while "opening" → HOME. Poses stay 0.05 rad inside the **real** limits, with the fingertips 3–25 cm above the table (checked with the **real** forward kinematics). Moves are smooth (minimum-jerk curves), every joint ≤ 0.5 rad/s. |
| State vs. action | Fake, realistic relationship | Action = commanded position. State = the same one step later, plus small noise, mimicking a follower arm lagging the leader. The gripper state lags and stops at 0.7 when closing on the "cube", as a real gripper blocked by an object would. |
| Images | Fake | A cartoon: brown table, red cube (moves while carried), blue gripper marker. Drawn at 640×480 and then put through the **real** preprocessing (`resize_with_pad` → 224×224). |
| Prompts | Fake | Two placeholder tasks, alternating, which tests that multiple tasks work |
| Size | 20 episodes, about 15 s each | Enough to exercise everything, small enough to be quick |

**Why `resize_with_pad` matters:** it shrinks the 640×480 image to fit 224×224 **without stretching it**, padding the leftover space with
black. Using the **same function** in the dataset and in the live client means training and serving images are processed identically.
A mismatch here, like a stretched image or swapped colour channels, is a classic silent failure.

### 5.8 `scripts/fake_policy_server.py`: a safe stand-in for the model
- It speaks the same WebSocket protocol as openpi's server, but returns **scripted** chunks: slow sine waves around HOME (j1 ±0.2 rad,
  j3 ±0.1 rad, 20 s period), with the gripper toggling every 6 s. It also simulates about 100 ms of model latency.
- **Safety by construction:**
  - Each step moves at most **0.2 rad/s** from the **observed** state, so it never jumps.
  - It always stays 0.05 rad inside the joint limits.
  - It starts smoothly even from the torque-off exit pose, where j2 is outside its limit.
- **Why:** in Phase 3 we test `vla_bridge` on the real arm. The bridge must be proven safe before any learned model drives the arm,
  and this server needs no GPU and can run on the robot PC.
- **Tested on 27 Oct:** the client received `(10, 5)` chunks moving gently from HOME (about 0.06 rad/s), with a 103 ms round trip.

### 5.9 The tests (`tests/`, 22 tests)

| File | Protects against |
|---|---|
| `test_gripper.py` | Inverted or mis-scaled gripper |
| `test_constants_match_robot.py` | Our copy of robot facts drifting from `low_level_controller` (skipped where that repo isn't present, e.g. on the server) |
| `test_fake_server.py` | The fake server ever exceeding the speed cap or the limits, over 200 chunks starting from the worst pose |
| `test_policy.py` | Wrong camera slots or masks; delta/absolute applied wrongly; actions not round-tripping back to absolute; tokenizer and padding shapes wrong |
| `test_train_configs.py` | Configs not registered, or drifting from the memory-tested model |
| `test_synthetic_dataset.py` | Fake motions leaving the limits or exceeding 0.5 rad/s |

**On the laptop:** 11 pass; the openpi-dependent ones skip. **On the server:** 15 pass; the 7 constants checks skip (no `low_level_controller` there).

---

## 6. Phase 2: what you do next, step by step

Do this on the server, inside tmux, with the environment active (`source scripts/env.sh`). First run `git pull`, so the server has
commit `5ca1ecb` or later.

1. **Run the tests**
   ```bash
   python -m pytest tests -q
   ```
   - *Expect:* everything passes. The constants test skips, because `low_level_controller` isn't on the server.
   - *If `test_policy.py` fails:* it's telling you our adapter disagrees with openpi. Send me the output.

2. **Make the fake dataset** (CPU, a few minutes)
   ```bash
   python scripts/make_synthetic_dataset.py --overwrite
   ```
   - *Expect:* 20 lines like `episode 3/20: 231 frames (15.4 s)`, then `✅ Wrote 20 episodes …` into `data/lerobot/local/omx_synthetic`.
   - *Why:* this is the first real test that our dataset format works with openpi's pinned LeRobot.

3. **Compute norm stats** (CPU)
   ```bash
   python scripts/compute_norm_stats.py pi05_omx_lora_debug
   ```
   - *Expect:* a progress bar, then `Writing stats to: …/assets/pi05_omx_lora_debug/local/omx_synthetic`.
   - *Why:* training refuses to start without norm stats. This is also the first time the dataset passes through our repack +
     `OMXInputs` + `DeltaActions`.

4. **Train the debug model** (GPU, roughly 15–30 min)
   ```bash
   bash scripts/gpu_check.sh train && python scripts/train.py pi05_omx_lora_debug --exp-name=dryrun --overwrite
   ```
   - *Expect:* weight loading, a few minutes of compiling at step 0, then a `loss=…` line every 10 steps, and checkpoints written at steps
     100 and 199 under `checkpoints/pi05_omx_lora_debug/dryrun/`.
   - **Note the s/it** from the progress bar. It gives the real-data training speed (the memory test's s/it is still missing).
   - The loss may go down a little but means nothing, because the data is fake.

5. **Serve our own checkpoint** (GPU, window 0)
   ```bash
   bash scripts/gpu_check.sh serve && python scripts/serve.py policy:checkpoint \
       --policy.config=pi05_omx_lora_debug --policy.dir=checkpoints/pi05_omx_lora_debug/dryrun/199
   ```

6. **Probe it** (window 1)
   ```bash
   python scripts/latency_probe.py --env omx
   ```
   - *Expect:* **`actions shape=(10, 5)`**, not DROID's `(15, 8)`. That proves our adapters, norm stats and checkpoint all work together.
   - **Never send these actions to the arm.** The model was trained on cartoons.

7. **Log it.** Send me the s/it, the latency numbers and any errors. I'll add them to `experiments.md`. **Phase 2 is then done.**

**If something breaks, likely causes:**
- A LeRobot API difference in the pinned version (step 2).
- The tokenizer needs a download (steps 3–4; it was cached by Phase 1).
- A port clash (step 5).

Each of these is a small fix. That's exactly why we test with fake data now.

---

## 7. Problems we hit and what they taught us

| Problem | Cause | Fix / lesson |
|---|---|---|
| Card was thought to be an A5000 | It's an A30 | Always check with `nvidia-smi`. The A30 has HBM memory with built-in ECC (no capacity loss), no display output, and MIG support. |
| GPU busy (21.6 GB used by another user) | Shared server | `gpu_check.sh` before every GPU job. Do CPU-only work while waiting. |
| `verify_env.py` said the GPU wasn't free when it was | The script's own JAX checks reserved 90% of the GPU **before** the GPU-free check ran | Check the GPU before JAX loads, and use `PREALLOCATE=false` for small tests. A tool can measure itself by accident. |
| Port 8000 already in use | Another service on the server | One setting (`VLA_POLICY_PORT=4000`) used by every script |
| First inference took 18 s | JAX compiles on first use | Warm up before timing. The robot bridge will make one warmup call before enabling the arm. |
| Released π0.5 outputs `(15, 8)` | It's fine-tuned for a 7-DoF Franka | No released checkpoint knows our arm, so fine-tuning is required |
| openpi has no π0.5 LoRA config | Upstream only ships π0 LoRA configs | We wrote our own, with the same structure as openpi's π0 LoRA recipe |
| openpi pins an old LeRobot | Upstream choice | Write our datasets with that exact version (the synthetic dataset already does) |

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **VLA** | Vision-language-action model: images + text + state → actions |
| **openpi** | Physical Intelligence's open-source code for π0/π0.5 |
| **Checkpoint** | Saved model weights at some training step |
| **Fine-tuning** | Continuing training of a pretrained model on new data |
| **LoRA** | Training small adapter matrices instead of all the weights |
| **Action chunk** | A predicted sequence of future actions, here 10 × 5 |
| **Horizon** | Number of steps in a chunk (10) |
| **Flow matching** | Generating actions by refining noise in a few steps (10) |
| **Norm stats** | Per-dimension statistics used to scale data to [−1, 1] |
| **Delta action** | Target minus current position |
| **Batch size** | Examples per training step (16) |
| **JIT / XLA** | JAX compiles computations into GPU code on first use |
| **bf16** | A 16-bit number format: half the memory of fp32, enough precision for frozen weights |
| **AdamW** | The optimizer. It keeps 2 extra values per trainable weight. |
| **EMA** | A smoothed running copy of the weights (disabled for memory) |
| **MIG** | Splitting one GPU into smaller virtual GPUs |
| **LeRobot dataset** | Hugging Face's on-disk format for robot demonstrations |
| **WebSocket / msgpack** | A persistent network connection / a compact binary message format |
| **p95** | 95% of measurements are at or below this value |
| **Teleoperation** | A person driving the robot (e.g. with a leader arm) to record demonstrations |
| **Sim-to-real gap** | Models trained on simulated images often fail on real camera images |
