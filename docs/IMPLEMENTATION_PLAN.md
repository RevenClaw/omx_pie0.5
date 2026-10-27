# π0.5 on OpenMANIPULATOR-X — Implementation Plan

Step-by-step plan for the `vla_layer` repository: set up openpi on the A30 server, fine-tune π0.5 on OpenMANIPULATOR-X (OM-X)
demonstrations, serve it, and drive the real arm through the existing ROS 2 controller stack (`low_level_controller`).

Source material:
- `docs/pie0.5.pdf` — the π0.5 paper (arXiv:2504.16054)
- `docs/Running π0.5 on an OpenMANIPULATOR-X_ All the Ways to Run It and an End-to-End Pipeline Plan.pdf` — route comparison
- `../low_level_controller/CLAUDE.md` and `../low_level_controller/docs/` — the robot side
- The openpi repository (`main`, checked 2026-10-26). See [Verified facts](#verified-facts-about-openpi-checked-2026-10-26).

Status: **Phase 0 repo files written (2026-10-26). Server steps not yet run.** openpi pinned at `215abfb` (2026-08-24).

---

## 1. Decisions

These are fixed for v1. If you change one, record it here, because most of them affect the dataset, training and serving at the same time.

| # | Decision | Value | Why |
|---|---|---|---|
| D1 | Framework | **openpi (JAX), LoRA fine-tune of `pi05_base`** | It is the only π0.5 training route that fits 24 GB. openpi PyTorch has no LoRA, full fine-tuning needs more than 70 GB, and LeRobot `pi05` needs expert-only training on 24 GB. |
| D2 | Serving | openpi `serve_policy`-style websocket server on the A30, port 8000 | Keeps the model separate from ROS. The robot PC runs only `openpi-client`. |
| D3 | Robot interface | New `vla_bridge` node in `low_level_controller/custom_package` → `/arm_controller/joint_trajectory` + `/gripper_controller/gripper_cmd` | The ROS-side design already recorded in `low_level_controller/CLAUDE.md`. No MoveIt and no `/ee_goal` in the streaming path. |
| D4 | Action space | Joint space, 5-D: `[j1, j2, j3, j4, gripper]`. Joints are absolute radians in the dataset and trained as **deltas** (`DeltaActions(make_bool_mask(4, -1))`). The gripper stays **absolute**. | A 4-DoF arm can't follow arbitrary EE poses, and IK in the loop adds failure modes. This follows the openpi ALOHA pattern. |
| D5 | State | 5-D: `[j1..j4 rad, gripper_norm]` from `/joint_states` | Same layout as the action |
| D6 | Gripper convention | `gripper_norm = (0.019 − q) / 0.029`, clipped to [0, 1]. **0 = open (q = 0.019 m), 1 = closed (q = −0.010 m).** | Matches common π datasets. Unit-tested in both directions (T2.2). |
| D7 | Cameras | `front` (fixed, whole workspace) → `base_0_rgb`; `wrist` → `left_wrist_0_rgb`; `right_wrist_0_rgb` is zeros with mask = False | Uses 2 of π0.5's 3 image slots |
| D8 | Image preprocessing | Recorded at 640×480 RGB. Stored in the dataset as **224×224 via `openpi_client.image_tools.resize_with_pad`**, the same function the client uses at run time. | Training and serving get identical preprocessing. Raw frames stay in the rosbags. |
| D9 | Control/record rate | **15 Hz** | Fast enough for a slow XM430 arm, with enough slack for network latency |
| D10 | Chunking | `action_horizon = 10` (as in `pi05_libero`); execute **5** steps, then re-query | Changing the horizon later means retraining |
| D11 | Normalization | openpi quantile norm stats computed **on our dataset only**, versioned with the dataset | OM-X is not in the pretraining mixture |
| D12 | Prompt | One fixed task string per task, for example `"pick up the red cube and place it in the bowl"` | The string must be identical across all episodes of a task |
| D13 | Environments | Server: openpi `uv` env (Python 3.11), optionally in Docker. Robot PC: ROS 2 Jazzy container plus `openpi-client`. **JAX never goes into the ROS container.** | Avoids the Python 3.11 vs 3.12 conflict and CUDA conflicts |
| D14 | openpi integration | **No fork.** openpi is a pinned git submodule. Our code is a separate package (`omx_pi05`) with thin entry scripts that build our `TrainConfig` and call openpi's functions. | Upstream stays easy to update. If the wrapper approach breaks, fall back to a fork branch that registers the configs in `config.py`. |

## 2. Verified facts about openpi (checked 2026-10-26)

Several details in the route-comparison PDF were written without web access. These were checked against openpi `main`:

- Requirements: more than 8 GB for inference, more than **22.5 GB for LoRA**, more than 70 GB for full fine-tuning. Tested on **Ubuntu 22.04 only**.
- `pyproject.toml`: `requires-python >= 3.11`, `jax[cuda12]==0.5.3`, and **`lerobot` pinned to git rev `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`**. This is an older LeRobot, so the dataset must be written with this exact version (see Phase 4).
- π0.5 checkpoints: `gs://openpi-assets/checkpoints/pi05_base` (for fine-tuning), `pi05_libero`, `pi05_droid`. They are cached in `~/.cache/openpi`, or under `$OPENPI_DATA_HOME` if set.
- **There is no π0.5 LoRA config upstream.** Existing LoRA configs are π0 and π0-FAST only (`pi0_libero_low_mem_finetune`, …). We write `pi05_omx_lora` ourselves on that model:
  `Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`, with `freeze_filter` taken from `get_freeze_filter()` and `ema_decay=None`.
- `Pi0Config` defaults: `action_dim=32`, `action_horizon=50`. With `pi05=True`, `max_token_len=200` and `discrete_state_input=True`, so the state is tokenized as text.
  Because `max_token_len` (200) is larger than π0's 48, activation memory is higher than in the π0 low-mem recipe. That is why we start at batch 16 rather than 32 (Phase 6).
- `pi05_libero` reference settings: `action_horizon=10`, `discrete_state_input=False`, batch 256, 30k steps, peak learning rate 5e-5, EMA 0.999. Those are full fine-tuning settings, not what we use.
- `LiberoInputs` zero-fills the right-wrist image and masks it (except for π0-FAST). `LiberoOutputs` slices off the padded action dimensions. Our `OMXInputs` and `OMXOutputs` are modelled on these.
- The server takes **unnormalized** state and normalizes on its side. Images must be uint8 HWC, made with `image_tools.resize_with_pad(img, 224, 224)` and `convert_to_uint8`.

Re-check these when you bump the openpi submodule.

---

## 3. Target layout of `vla_layer/`

```
vla_layer/
├── CLAUDE.md                     # context for Claude Code sessions (Phase 0)
├── README.md                     # quickstart: setup → smoke test → train → serve
├── docs/
│   ├── IMPLEMENTATION_PLAN.md    # this file
│   ├── pie0.5.pdf, Running π0.5 … .pdf
│   ├── conventions.md            # D4–D8 written out as the single source of truth
│   └── experiments.md            # log: dataset hash, config, checkpoint, success rate
├── third_party/openpi/           # git submodule, pinned commit
├── omx_pi05/                     # our Python package, installed into openpi's uv env
│   ├── constants.py              # joint names, limits, gripper map, camera keys, FPS, HORIZON
│   ├── gripper.py                # metres ⇄ [0,1]
│   ├── policy.py                 # OMXInputs, OMXOutputs, make_omx_example()
│   ├── data_config.py            # LeRobotOMXDataConfig (repack + delta transforms)
│   └── train_configs.py          # pi05_omx_lora, pi05_omx_lora_debug, get_config(name)
├── scripts/
│   ├── gpu_check.sh              # refuse to start if < 23 GB free on the GPU
│   ├── setup_server.sh           # uv, submodule, uv sync, install omx_pi05, openpi-client
│   ├── download_checkpoints.py   # pre-fetch pi05_base / pi05_droid / pi05_libero
│   ├── serve_pretrained.sh       # zero-shot pi05_droid / pi05_libero (smoke test)
│   ├── latency_probe.py          # client: N infer() calls on dummy obs, latency stats
│   ├── memory_test.py            # LoRA fit test on fake data: peak GPU memory per batch size
│   ├── make_synthetic_dataset.py # fake OM-X LeRobot dataset for testing the pipeline
│   ├── convert_rosbag_to_lerobot.py
│   ├── inspect_dataset.py        # stats, plots, contact sheet of frames per episode
│   ├── compute_norm_stats.py     # wrapper → openpi norm stats for our config
│   ├── train.py                  # wrapper → openpi training loop for our config
│   ├── serve.py                  # wrapper → load our checkpoint + WebsocketPolicyServer
│   └── replay_test.py            # open-loop: dataset frames → server → plot vs recorded
├── docker/                       # optional: Dockerfile (Ubuntu 22.04 + CUDA + uv), compose
├── tests/                        # pytest: gripper map, transforms round-trip, policy I/O shapes
├── data/        (gitignored)     # rosbags, LeRobot datasets
└── checkpoints/ (gitignored)
```

Robot-side code goes into `low_level_controller/custom_package`, which keeps that repo's isolation policy (see Phase 3 and Phase 4):
`vla_bridge.py`, a recorder launch file, and a leader-arm teleop node.

---

## 4. Phases

Each phase lists its **tasks**, its **exit criteria** (don't start the next phase until they pass), and where the code lives.
`[S]` = A30 server, `[R]` = robot PC / ROS container.

### Phase 0 — Repository and server environment `[S]`

| Task | Detail |
|---|---|
| T0.1 | Check the server: `lsb_release -a` (Ubuntu 22.04 is the tested OS; 24.04 is likely fine with `uv` and otherwise needs Docker), free disk (**≥ 150 GB** for checkpoints, datasets and runs), and whether you have `sudo` and Docker with `nvidia-container-toolkit`. |
| T0.2 | Free the GPU. `scripts/gpu_check.sh train\|serve\|<MiB>` exits non-zero if free memory is below 23 000 MiB (train) or 9 000 MiB (serve), or if MIG is enabled, and lists the processes on the GPU. Every train or serve script calls it first. Keep MIG **disabled**. |
| T0.3 | `git init`, add openpi as a submodule (`git submodule add https://github.com/Physical-Intelligence/openpi third_party/openpi`, then `git submodule update --init --recursive`), and **pin the commit**. Add `.gitignore` entries for `data/`, `checkpoints/`, `.venv/`, `wandb/`. |
| T0.4 | `setup_server.sh`: install `uv`, then in `third_party/openpi` run `GIT_LFS_SKIP_SMUDGE=1 uv sync` and `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`, then install `omx_pi05` into the same venv. `OPENPI_DATA_HOME` goes in `.env` (see `.env.example`); `scripts/env.sh` loads it and activates the venv. |
| T0.5 | `download_checkpoints.py`: pre-fetch `pi05_base`, `pi05_droid` and `pi05_libero` with `openpi.shared.download.maybe_download`. |
| T0.6 | Write `vla_layer/CLAUDE.md` (layout, decisions D1–D14, commands, "keep docs in sync"). |

**Exit:** `python scripts/verify_env.py` passes: JAX sees the A30, a bf16 matmul runs, openpi / openpi-client / lerobot / omx_pi05 import, disk ≥ 150 GB, checkpoints cached, and `gpu_check.sh train` passes.

### Phase 1 — Smoke test with released checkpoints `[S]` + `[R]`

This checks the plumbing only. Released checkpoints don't know OM-X, so their actions are meaningless on our arm.

| Task | Detail |
|---|---|
| T1.1 | `serve_pretrained.sh`: `uv run scripts/serve_policy.py --env=DROID` (or `policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid`). Watch `nvidia-smi`: memory use should be about 8–10 GB. |
| T1.2 | `latency_probe.py`: build a dummy DROID-shaped observation, call `WebsocketClientPolicy(host, 8000).infer()` 100 times, and print p50/p95/p99 latency. Run it **on the server** (model time) and **from the robot PC** (model plus network). |
| T1.3 | Network: if port 8000 isn't reachable from the robot PC, use an SSH tunnel (`ssh -N -L 8000:localhost:8000 user@server`). Record which one you use in `README.md`. |
| T1.4 | (Optional) LIBERO sim eval with `pi05_libero`, following `third_party/openpi/examples/libero` (headless MuJoCo/EGL on the server). Confirms the install against the published numbers. |
| T1.5 | Install `openpi-client` in the ROS container (`pip install -e third_party/openpi/packages/openpi-client`; it needs only numpy, msgpack and websockets) and run `latency_probe.py` from there. |

**Exit:** p95 round trip from the ROS container is written in `experiments.md`. Target: **< 150 ms**, which is ≈ 2 steps at 15 Hz.

### Phase 2 — OM-X integration code and a synthetic dry run `[S]`

This builds and tests the whole openpi pipeline before any real data exists, so bugs in it show up early.

| Task | Detail |
|---|---|
| T2.1 | `constants.py`: `JOINT_NAMES = [joint1..joint4]`, `GRIPPER_JOINT = gripper_left_joint`, joint limits from `low_level_controller/CLAUDE.md` (j1 [−π, π], j2 [−1.82, 1.5], j3 [−1.5, 1.55], j4 [−1.7, 1.97]), gripper [−0.010, 0.019] m, `FPS = 15`, `ACTION_HORIZON = 10`, `CAMERA_KEYS`, `HOME`. |
| T2.2 | `gripper.py`: `metres_to_norm` and `norm_to_metres` (D6). Tests check both directions, the endpoints, clipping, and that open → 0. |
| T2.3 | `policy.py`: **`OMXInputs`** maps `observation/front_image` → `base_0_rgb`, `observation/wrist_image` → `left_wrist_0_rgb`, and fills `right_wrist_0_rgb` with zeros. Image masks are front True, wrist True, right False. It passes through `state` (5,), `actions` when present, and `prompt`. It reuses `_parse_image` behaviour (float → uint8, CHW → HWC). **`OMXOutputs`** returns `actions[:, :5]`. `make_omx_example()` returns a random observation for tests. |
| T2.4 | `data_config.py`: `LeRobotOMXDataConfig(DataConfigFactory)`. The repack maps dataset keys `observation.images.front`, `observation.images.wrist`, `observation.state`, `action`, `prompt` → policy keys. Data transforms are `OMXInputs`/`OMXOutputs`, plus `DeltaActions(make_bool_mask(4, -1))` on inputs and `AbsoluteActions(...)` on outputs. Model transforms come from `ModelTransformFactory()(model_config)`. Uses `prompt_from_task=True`. |
| T2.5 | `train_configs.py`: **`pi05_omx_lora`** with `model=Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`, `freeze_filter=` that same config's `.get_freeze_filter()`, `ema_decay=None`, `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`, `batch_size=16`, `num_train_steps=30_000`, `save_interval=5_000`, `keep_period=5_000`, and the default cosine LR schedule (warmup 1k). Also add **`pi05_omx_lora_debug`**: 200 steps, batch 8, `save_interval=100`, wandb off. |
| T2.6 | Wrapper scripts. `train.py` loads openpi's `scripts/train.py` as a module and calls `main(get_config(name))`. `compute_norm_stats.py` does the same with openpi's norm-stats routine. `serve.py` calls `openpi.policies.policy_config.create_trained_policy(cfg, ckpt_dir)` and runs it through `openpi.serving.websocket_policy_server.WebsocketPolicyServer(policy, port=8000)`. If upstream internals make this awkward, use the D14 fallback (fork branch). |
| T2.7 | `make_synthetic_dataset.py`: writes about 20 fake episodes (random 224×224 images, smooth sinusoidal joint trajectories within limits, gripper steps) **using openpi's pinned `lerobot`**, with the exact feature schema from Phase 4. |
| T2.8 | Dry run: synthetic dataset → norm stats → `pi05_omx_lora_debug` training → `serve.py` → `latency_probe.py` with an OM-X-shaped observation, which must return a `(10, 5)` array. |
| T2.9 | **Memory test on the A30** (`scripts/memory_test.py`: the real pi05_base LoRA model on openpi's FakeDataset, so it can run before T2.1–T2.8 exist): run `pi05_omx_lora` for 100–300 steps at batch 16 with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. Record peak memory and steps/s. Then try 32. If it runs out of memory, try 8, then raise the fraction to 0.95. Write the chosen batch size into the config. |
| T2.10 | `tests/`: transforms round trip (absolute → delta → absolute gives back the original), `OMXInputs` output shapes and masks, and that state and action stay 5-D before padding. |

**Exit:** the full pipeline runs on synthetic data, the batch size is fixed from the memory test, and the time per 1k steps is known.

### Phase 3 — `vla_bridge` on the robot `[R]` (in `low_level_controller/custom_package`)

The bridge is what keeps the arm safe. Build it and test it against a **fake policy server** before any model is involved.

| Task | Detail |
|---|---|
| T3.1 | `scripts/fake_policy_server.py` (in `vla_layer`, no GPU needed): a websocket server with the same msgpack protocol (use `openpi_client`'s `msgpack_numpy` and the openpi server class with a stub policy). It returns `(10, 5)` chunks of a slow sinusoid around HOME plus a gripper toggle. |
| T3.2 | `vla_bridge.py` inputs: subscribe to `/joint_states` and both camera topics, keeping the latest message and its stamp. On each **inference tick**: build `{"observation/front_image", "observation/wrist_image"}` (uint8 224×224 via `resize_with_pad`, **RGB**), `"observation/state"` (5,) float32 in the D5/D6 convention, and `"prompt"`. Call `infer()` on a worker thread so the control timer never blocks. |
| T3.3 | Chunk buffer: tag each chunk with the observation time. When it arrives, **drop the actions whose time has already passed** (latency compensation). Re-query when `remaining ≤ ceil(latency·fps) + 1`. In v1, a new chunk simply replaces the old one; add a linear cross-fade over 2–3 steps only if the joins look jerky. |
| T3.4 | Output: publish the executable slice as **one multi-point `JointTrajectory`** (`time_from_start = (k+1)/fps`) on `/arm_controller/joint_trajectory`. The JTC splines between points and replaces the old trajectory. Send the gripper as `GripperCommand` on `/gripper_controller/gripper_cmd`, only when the hysteresis state changes (close > 0.6, open < 0.4), mapped back with `norm_to_metres`. |
| T3.5 | Safety layer, applied to every point before it is published: clamp to joint limits minus 0.05 rad; limit per-step Δq to `max_joint_vel / fps` (start at **0.3 rad/s**, the "25%" stage); workspace box check with `ee_commander.forward_kinematics` (z ≥ min_z). Watchdog: hold position if `/joint_states` is older than 0.1 s, an image is older than 0.2 s, or no chunk arrives within 2× the expected latency. Add a `/vla_bridge/estop` topic. **Start disabled**; it becomes active only after `/vla_bridge/enable` (`std_srvs/SetBool`). |
| T3.6 | `shadow` parameter: run the full loop but publish to `/vla_bridge/preview` (`JointTrajectory`) and to an RViz marker instead of the controller. |
| T3.7 | Launch with `init_position:=false`. Document it in `low_level_controller/docs/commands_guide.md` and the architecture manual, and log it in `openmanipulator_changes_summary.md`. Mark it "not yet tested on hardware" in `known_issues.md` until it has been run. |

**Exit:** with the fake server, the real arm follows the sinusoid smoothly at 0.3 rad/s. Killing the server stops the arm (hold). The e-stop topic works. No joint2 overload (`0x20`).

### Phase 4 — Data collection setup `[R]`

| Task | Detail |
|---|---|
| T4.1 | **Fix known_issues #1** (gripper path in `joint_commander`) and confirm that `GripperCommand` works on hardware. Teleop recording depends on it. |
| T4.2 | Cameras: one fixed front camera (C920 / RealSense in RGB mode) covering the whole workspace, and one small USB wrist camera on link 5. Run them with `usb_cam` or `v4l2_camera` at 640×480, 30 fps, **fixed exposure and white balance**. Mark the front camera's position on the table; it must never move. |
| T4.3 | Teleop, best option first: a second OM-X as a torque-off leader (1:1 mapping) → GELLO-style leader → gamepad/keyboard. The leader node publishes the commanded joint positions, `/leader/joint_states`. **The action recorded at time t is the leader position at t.** |
| T4.4 | Safety: a hardware e-stop that cuts motor power, and the Dynamixel Current Limit set below the factory value (EEPROM; use the same procedure as in `joint_offset_investigation.md`). |
| T4.5 | Recorder launch: rosbag2 (mcap) of `/joint_states`, `/leader/joint_states`, `/arm_controller/controller_state`, both `image_raw` + `camera_info`, and `/episode` (a `std_msgs/String` JSON with `{task, episode_id, success}` published on start/stop). Optional extras for the later surprise monitor: Present Current and Velocity, if `dynamixel_hardware_interface` exposes current as a state interface (check its config; never call `get_dxl_data` on a running arm, issue #8). |
| T4.6 | Episode protocol: start every episode from HOME; vary the object placement on a marked grid; use one task string; trim idle frames at both ends; tag failed demos as `success=false` and exclude them. Aim for 10–20 s per episode. |

**Exit:** 5 test episodes recorded, and the bags play back with every topic at the expected rate.

### Phase 5 — Dataset conversion and validation `[S]`

| Task | Detail |
|---|---|
| T5.1 | `convert_rosbag_to_lerobot.py`: read bags with the pure-Python **`rosbags`** library (no ROS install needed on the server). Resample every stream to a 15 Hz grid using the latest previous message by header stamp, and **reject frames whose skew is more than half a period**. Images go BGR → RGB → `resize_with_pad(224)`. Write with **openpi's pinned `lerobot`**: `LeRobotDataset.create(repo_id="local/omx_<task>", fps=15, robot_type="openmanipulator_x", features=...)`, then `add_frame()` per step and `save_episode()` per episode. |
| T5.2 | Features: `observation.images.front` and `observation.images.wrist` are (224, 224, 3) image; `observation.state` is (5,) float32; `action` is (5,) float32 (leader joints in rad plus leader gripper normalized); `task` is the prompt. Optional: `observation.current` (5,) and `observation.velocity` (5,). |
| T5.3 | `inspect_dataset.py`: per-dimension min/max/quantiles, state vs action plots for each episode, a contact sheet of front and wrist frames at t = 0, mid and end (catches swapped cameras or flipped images), and a gripper histogram (should be bimodal near 0 and 1). |
| T5.4 | Dataset hash (episode list + converter git SHA). Record it in `experiments.md`; the norm stats are tied to it. |
| T5.5 | (For Phase 5b) Export a copy for current LeRobot, either with LeRobot's v2.1 → v3.0 converter or by writing it a second time. |

**Exit:** **50 good demos** of one task convert cleanly and look right in `inspect_dataset.py`.

### Phase 5b — ACT baseline (recommended, separate LeRobot env) `[S]`

Train ACT (and then SmolVLA, if you want) with `lerobot-train` on the same data in a **third** environment. If ACT reaches **≥ 50%** on the task, the data pipeline is probably sound. If π0.5 then fails, the problem is in openpi config or serving, not in the data. Skip this only if time is very short.

### Phase 6 — π0.5 LoRA fine-tune `[S]`

| Task | Detail |
|---|---|
| T6.1 | Point `pi05_omx_lora` at `local/omx_<task>` and run `scripts/gpu_check.sh && uv run scripts/compute_norm_stats.py --config-name pi05_omx_lora`. **Re-run it every time the data changes.** |
| T6.2 | Train headless, inside `tmux`: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_omx_lora --exp-name=omx_<task>_v1`. Turn on wandb to log the loss and grad norm. |
| T6.3 | Checkpoints every 5k steps, each with its `assets/` (norm stats). Keep all of them, because the best one is chosen on the robot, not by loss (flow-matching loss correlates poorly with success). |
| T6.4 | If it runs out of memory: lower the batch (16 → 8 → 4), set the memory fraction to 0.95, and make sure nothing else is on the GPU (`gpu_check.sh`). |
| T6.5 | Log the run in `experiments.md`: dataset hash, openpi commit, config, batch, steps, wall time. |

**Exit:** checkpoints at 5k/10k/…/30k exist, and the loss has plateaued.

### Phase 7 — Offline validation `[S]`

| Task | Detail |
|---|---|
| T7.1 | `replay_test.py`: for held-out episodes, send each recorded frame's observation (exactly as the bridge would build it) to `serve.py` and plot predicted chunk[0..4] against the recorded actions for every dimension. |
| T7.2 | What this catches: an offset means a normalization or delta/absolute bug; an inverted gripper means the D6 convention is wrong; garbage output means swapped camera keys or BGR. Also save the server-side preprocessed 224×224 inputs once and look at them. |
| T7.3 | Compare checkpoints by replay error as a **first filter only**. |

**Exit:** predictions track the recorded actions on held-out episodes, and the gripper switches at the right moments.

### Phase 8 — Deployment on the robot `[S]` + `[R]`

| Step | Detail |
|---|---|
| T8.1 | Serve the chosen checkpoint: `uv run scripts/serve.py --config pi05_omx_lora --dir checkpoints/pi05_omx_lora/omx_<task>_v1/<step>`. |
| T8.2 | **Shadow mode** (`vla_bridge shadow:=true`): the arm stays still while you move the object and watch the preview trajectory in RViz. |
| T8.3 | Live at 0.3 rad/s with a hand on the e-stop. Then 0.5, then the speed used in the demos. |
| T8.4 | Tune `replan_every` (3/5/8) and the gripper hysteresis. |

**Exit:** the first successful autonomous grasp-and-place.

### Phase 9 — Evaluation and iteration

- Protocol: 20 trials per task on a fixed set of randomized object placements. Record success, time to success and interventions. Log in `experiments.md`.
- Compare checkpoints (10k/20k/30k) and against ACT. Ablations: wrist camera on/off, `replan_every`.
- Iterate: collect more demos where it fails (100–200 for robustness to placement), recompute norm stats, retrain.

### Phase 10 — Later (only after Phase 9 works)

1. Multitask and language: 2–3 tasks with distinct prompts.
2. LeRobot `pi05` (`train_expert_only`, bf16, gradient checkpointing) for async inference and **RTC** chunk joins.
3. Surprise monitor in `vla_bridge`: predicted vs measured current and position. On a large residual, flush the buffer, hold, and re-prompt.
4. External high-level planner (a VLM that writes π0.5's prompt). openpi exposes only the action head, not subtask prediction.
5. RL post-training (RLinf) or a full fine-tune on a rented 80 GB GPU.

---

## 5. Timeline (rough, one person)

| Week | Phases |
|---|---|
| 1 | 0, 1, 2 (synthetic dry run and A30 memory test) |
| 2 | 3 (bridge with the fake server on the real arm), start 4 |
| 3–4 | 4, 5 (hardware for cameras and leader, 50 demos, converter) |
| 5 | 5b (ACT baseline) |
| 6–7 | 6, 7, 8 |
| 8+ | 9, 10 |

Phases 2 and 3 need no data and can run in parallel with the Phase 4 hardware work.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Norm stats don't match the data, or the wrong assets are served | Recompute after every data change. Stats are stored in the checkpoint and tied to the dataset hash. |
| Camera keys swapped, BGR, flipped images | The same `resize_with_pad` path on both sides (D8). Contact sheet (T5.3) and replay test (T7.1). |
| Gripper inverted or in the wrong units | D6 with unit tests (T2.2) and the gripper histogram (T5.3) |
| Delta transform applied in training but not undone in serving | Both come from the same `LeRobotOMXDataConfig`. Round-trip test (T2.10). |
| LeRobot version mismatch (openpi pins an old rev) | Write the dataset with openpi's pinned `lerobot` (T5.1). A separate export for current LeRobot (T5.5). |
| Out of memory on 24 GB | Memory test in Phase 2, `gpu_check.sh`, batch fallbacks, headless training |
| **Shared GPU** (another job held 21.6 GB on 2026-10-26) | `gpu_check.sh` refuses to start if the GPU is busy. Agree on time slots for training. |
| Network latency or jitter between the server and the robot | Latency compensation and early re-query (T3.3), watchdog hold (T3.5), latency measured in Phase 1 |
| XM430 overload / joint2 `0x20` | Δq velocity clamp, lower current limit, start at 0.3 rad/s, HOME start, hardware e-stop |
| Idle frames make the policy hesitate | Trim idle frames (T4.6) |
| openpi upstream changes | Pinned submodule. §2 is re-checked on every bump. |

## 7. Open questions (answer before or during Phase 0)

1. Server OS and whether you have `sudo` or Docker. This decides between plain `uv` and Docker.
2. Is the robot PC on the same network as the A30 server? Is port 8000 reachable, or is an SSH tunnel needed?
3. Is a second OM-X available as a leader arm? If not, GELLO or a gamepad.
4. Which camera models, and how will the wrist camera be mounted?
5. The first task and its exact prompt string.
6. Who else uses the A30, and when can it be reserved for multi-hour training runs?
