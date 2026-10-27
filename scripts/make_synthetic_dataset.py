"""Phase 2: write a FAKE OM-X LeRobot dataset to test the pipeline end to end before real demos exist.

    source scripts/env.sh
    python scripts/make_synthetic_dataset.py                 # 20 episodes → $HF_LEROBOT_HOME/local/omx_synthetic
    python scripts/make_synthetic_dataset.py --episodes 5 --overwrite

What is real: the dataset format (feature names, shapes, dtypes, fps, LeRobot version = openpi's pinned one), the joint
limits, HOME pose, kinematics, gripper range and image preprocessing (640×480 → resize_with_pad → 224×224).
What is fake: the motions (pick-and-place-like joint trajectories between random reachable poses, ≤ 0.5 rad/s), the
images (flat cartoon of table, cube and gripper), and the prompts. A model trained on this learns nothing useful.
"""

import dataclasses
import shutil

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from openpi_client import image_tools

from omx_pi05 import constants as C
from omx_pi05.kinematics import forward_kinematics

TASKS = ("pick up the cube and place it to the left", "pick up the cube and place it to the right")
CAM_H, CAM_W = 480, 640  # recorded camera resolution (plan T4.2)
MAX_VEL = 0.5  # rad/s, peak joint speed of the fake motions
KEYFRAME_BOX = np.array([[-0.8, 0.8], [-0.6, 0.6], [-0.6, 0.8], [0.0, 1.2]])  # sampling box for j1..j4, rad


@dataclasses.dataclass
class Args:
    repo_id: str = "local/omx_synthetic"
    episodes: int = 20
    seed: int = 0
    overwrite: bool = False


def features() -> dict:
    image = {"dtype": "image", "shape": (C.IMAGE_SIZE, C.IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    vec = {"dtype": "float32", "shape": (C.STATE_DIM,), "names": [*C.ARM_JOINTS, "gripper"]}
    return {C.DS_FRONT: image, C.DS_WRIST: dict(image), C.DS_STATE: vec, C.DS_ACTION: dict(vec)}


def sample_pose(rng: np.random.Generator) -> np.ndarray:
    """Random joint pose inside the limits whose fingertips are 3–25 cm above the table."""
    lo, hi = C.joint_limits_array(C.JOINT_LIMIT_MARGIN)
    while True:
        q = rng.uniform(KEYFRAME_BOX[:, 0], KEYFRAME_BOX[:, 1])
        if np.all(q > lo) and np.all(q < hi) and 0.03 <= forward_kinematics(q)[2] <= 0.25:
            return q


def min_jerk(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)[:, None]
    return q0 + (q1 - q0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def segment(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    # Min-jerk peak speed is 1.875·Δ/T, so T = 1.875·Δmax/MAX_VEL keeps every joint under MAX_VEL.
    duration = max(1.0, 1.875 * float(np.max(np.abs(q1 - q0))) / MAX_VEL)
    return min_jerk(q0, q1, int(np.ceil(duration * C.FPS)) + 1)  # n−1 steps of 1/FPS span ≥ duration


def make_episode(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns commanded arm joints (T,4), commanded gripper (T,) in [0,1], and the "grasped" flag (T,)."""
    home = np.array(C.HOME)
    pick, place = sample_pose(rng), sample_pose(rng)
    hold = int(0.7 * C.FPS)
    parts = [  # (joints, gripper command, grasped)
        (segment(home, pick), 0.0, False),
        (np.repeat(pick[None], hold, 0), 1.0, False),  # close
        (segment(pick, place), 1.0, True),
        (np.repeat(place[None], hold, 0), 0.0, True),  # open
        (segment(place, home), 0.0, False),
    ]
    joints = np.concatenate([p[0] for p in parts])
    grip = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
    grasped = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
    return joints, grip, grasped


def to_pixels(x: float, y: float) -> tuple[int, int]:
    # Top-down front camera: x (forward) → image up, y (left) → image left. 0.4 m × 0.53 m field of view.
    return int(CAM_H - x / 0.4 * CAM_H), int(CAM_W / 2 - y / 0.53 * CAM_W)


def draw_square(img: np.ndarray, centre: tuple[int, int], half: int, color) -> None:
    r, c = centre
    img[max(r - half, 0) : max(r + half, 0), max(c - half, 0) : max(c + half, 0)] = color


def render(q: np.ndarray, cube_xy: tuple[float, float], gripper_norm: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, _ = forward_kinematics(q)
    front = np.full((CAM_H, CAM_W, 3), (150, 120, 90), dtype=np.uint8)  # table
    draw_square(front, to_pixels(*cube_xy), 18, (200, 30, 30))  # red cube
    draw_square(front, to_pixels(x, y), int(10 + 60 * z), (30, 60, 200))  # gripper, bigger when higher
    wrist = np.full((CAM_H, CAM_W, 3), (150, 120, 90), dtype=np.uint8)
    dx, dy = cube_xy[0] - x, cube_xy[1] - y
    draw_square(wrist, (int(CAM_H / 2 - dx * 1500), int(CAM_W / 2 - dy * 1500)), int(30 / max(z, 0.03)), (200, 30, 30))
    fingers = int(40 + 200 * (1 - gripper_norm))  # finger gap shrinks as the gripper closes
    draw_square(wrist, (CAM_H - 30, CAM_W // 2 - fingers), 25, (60, 60, 60))
    draw_square(wrist, (CAM_H - 30, CAM_W // 2 + fingers), 25, (60, 60, 60))
    # Same preprocessing as the real client and converter (plan D8).
    return tuple(image_tools.resize_with_pad(im, C.IMAGE_SIZE, C.IMAGE_SIZE) for im in (front, wrist))


def main(args: Args) -> None:
    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"❌ {out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="openmanipulator_x",
        fps=C.FPS,
        features=features(),
        image_writer_threads=8,
        image_writer_processes=2,
    )
    rng = np.random.default_rng(args.seed)
    frames = 0
    for ep in range(args.episodes):
        cmd_joints, cmd_grip, grasped = make_episode(rng)
        task = TASKS[ep % len(TASKS)]
        # Follower lags the leader by one step plus sensor noise; the gripper closes with a lag and stops at 0.7
        # when it closes on the cube (as a real gripper blocked by an object would).
        state_joints = np.vstack([cmd_joints[:1], cmd_joints[:-1]]) + rng.normal(0, 0.002, cmd_joints.shape)
        state_grip, g = np.empty_like(cmd_grip), 0.0
        for t, target in enumerate(cmd_grip):
            g += (min(target, 0.7) - g) * (1 - np.exp(-1 / (0.2 * C.FPS)))
            state_grip[t] = g
        cube =forward_kinematics(cmd_joints[np.argmax(grasped)])[:2]  # the cube sits at the pick pose
        for t in range(len(cmd_joints)):
            if grasped[t]:
                cube = forward_kinematics(state_joints[t])[:2]  # carried
            front, wrist = render(state_joints[t], cube, state_grip[t])
            dataset.add_frame(
                {
                    C.DS_FRONT: front,
                    C.DS_WRIST: wrist,
                    C.DS_STATE: np.append(state_joints[t], state_grip[t]).astype(np.float32),
                    C.DS_ACTION: np.append(cmd_joints[t], cmd_grip[t]).astype(np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
        frames += len(cmd_joints)
        print(f"   episode {ep + 1}/{args.episodes}: {len(cmd_joints)} frames ({len(cmd_joints) / C.FPS:.1f} s) '{task}'")
    print(f"✅ Wrote {args.episodes} episodes, {frames} frames at {C.FPS} Hz to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
