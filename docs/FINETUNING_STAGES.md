# Fine-tuning Stages: from pick-and-place to general commands

**Status (2026-11-05): agreed direction; v1 tooling built** (step-by-step: `low_level_controller/docs/finetuning_procedure.md`).
How the data is produced: `DATASET_PLAN.md`. Recording procedure and tools:
`PHASE4_GUIDE.md`, `PHASE5_GUIDE.md`.

---

## 1. The premise: high-level commands are sequences of pick-and-place

Most tabletop commands our OM-X can carry out are **rearrangement tasks**: the goal is a new arrangement of objects. Each one
breaks down into a sequence of small commands of the form **"pick up X and place it at Y"**:

| High-level command | Breakdown into pick-and-place |
|---|---|
| "sort the cubes by colour" | for each cube: pick(cube) → place(bin of its colour) |
| "clear the table" | for each object: pick(object) → place(bin) |
| "stack the blue cube on the red one" | pick(blue) → place(on top of red) |
| "put the red cube left of the bin" | pick(red) → place(a point left of the bin) |
| "swap the red and blue cubes" | pick(red) → place(free spot) → pick(blue) → place(red's old spot) → pick(red) → place(blue's old spot) |

The "place at Y" part carries a **mode** (in a bin, on top of an object, at a point), and a sequence may need re-detection after
each step (e.g. "keep sorting until the table is clear"). With a 4-DOF arm and a simple gripper, this covers almost everything
the robot can realistically do.

**So the natural first fine-tuning stage is to teach the pretrained π0.5 the pick-and-place actions of our robot.** π0.5 already
understands language and objects from pretraining; what it has never seen is *this* arm: its joints, its gripper, its camera
view. Once it can reliably execute one pick-and-place command, more general commands can be built on top (§4).

---

## 2. Stage 1 (v1): pick-and-place on our OM-X

| | v1 contains |
|---|---|
| Skill | One: pick up a cube and place it in a bin |
| Robot | Only our OM-X (its joints, gripper, front camera view) |
| Data source | Scripted demos: YOLO finds the cube and bin, IK + smooth motion executes, with small random variations |
| Variety within the skill | Cube positions over a grid, 2–3 cube colours, bins at their tagged spots |
| Prompts | Template: "pick up the {red/blue/green} cube and place it in the {bin}" |
| Size | Pilot of 5–10 episodes, then ~200–300 |
| Purpose | Prove the whole chain on the real robot (record → convert → train → serve → bridge → the arm does the task), and find out what the model gets wrong |

---

## 3. What v1 is limited to

A fine-tuned VLA does well what it was shown on this robot, and only partly beyond that.

| Kind of generalization | After v1? | Why |
|---|---|---|
| New cube **positions** on the table | ✅ yes | Positions vary in the data (grid) |
| New **wording** of the same command | 🟡 partly | Language comes from pretraining; reliable only after paraphrases (v2) |
| New **combinations** (a colour → a bin never paired in the data) | 🟡 sometimes | Possible if each part appears in the data; must be tested |
| New **objects or colours** of a similar kind | 🟡 maybe | Pretraining knows them, but grasping them on the OM-X was never shown; test, don't assume |
| **Recovery** after a slip or near miss | 🟡 weak | Scripted demos rarely show recoveries; random variation helps a little |
| **Other skills** (stacking, placing at a point, pushing) | ❌ no | A motion never demonstrated on this arm isn't learned |
| **Multi-step** commands ("sort all the cubes") | ❌ not by itself | Needs a planner (v3) or long-horizon demos (v4) |
| **Rotating** objects about the vertical axis | ❌ never | Hardware: the gripper's yaw always follows the base direction (4-DOF) |
| Non-pick-and-place actions (drawers, buttons, pouring, wiping, cloth) | ❌ out of scope | Need contact/force skills and other primitives |

---

## 4. Next stages

- **v2: robust pick-and-place.**
  - LLM-paraphrased prompts attached to the same episodes (varied wording, little or no extra recording).
  - Leader-arm demos targeted at v1's failures (recoveries after slips and near misses, hard positions), ~20–30 per failure type.
  - More variety: distractor objects on the table, more object types and colours.
- **v3: more primitives plus an LLM planner.**
  - New primitives, each with ~50–200 demos: `place on top`, `place at a position`, maybe `push`.
  - An LLM breaks high-level commands into these primitives and sends them to the VLA one at a time, re-detecting the scene
    after each step. The VLA executes; the LLM plans. Each plan is validated, and the final scene is checked against its goal.
- **v4 (optional): long-horizon demos.**
  - Whole multi-step episodes recorded with the high-level prompt and per-step sub-task labels, so π0.5's own two-level reasoning
    (predict the next sub-task, then act) learns the decomposition. Only if the v3 planner approach falls short; it costs more data.
- **Every stage:** retrain on the combined data, evaluate on the same fixed trials (including unseen positions and wording), and
  compare against the modular LLM + YOLO + IK pipeline (`DATASET_PLAN.md` §9). Results go in `experiments.md`.
