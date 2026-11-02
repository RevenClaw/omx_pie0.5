"""Write FAKE OM-X episode recordings as real rosbag2 (mcap) directories, in the format the robot-side recorder writes.

    source scripts/env.sh
    python scripts/make_synthetic_bags.py                          # 5 episodes → data/rosbags/synthetic/episode_0000 ...
    python scripts/make_synthetic_bags.py --out /tmp/bags --episodes 3 --gap-episode 1 --fail-episode 2

For testing scripts/convert_rosbag_to_lerobot.py without the robot. Each episode directory holds the bag (metadata.yaml +
.mcap) and metadata.json ({"task", "success", ...}). Streams, with realistic rates, header stamps and delays:
  /teleop/action                 30 Hz   JointState joint1..4 + "gripper" (0 open..1 closed): HOME → pick → place → HOME,
                                         with 1 s of idle before and after (the converter trims it)
  /joint_states                  100 Hz  JointState joint1..4 + gripper_left_joint (m): the action delayed by 0.1 s
  /camera/front/color/image_raw  15 Hz   rgb8 640×480: a blue square at the fingertip position (bigger when higher)
  /camera/wrist/image_raw        15 Hz   rgb8 640×480: grey "fingers" whose gap follows the gripper
--gap-episode drops 0.5 s of front frames in that episode (the converter must reject it); --fail-episode marks one as failed.
Needs only numpy and rosbags (pinned in scripts/setup_server.sh), no ROS and no lerobot.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from omx_pi05 import constants as C
from omx_pi05.dataset import ACTION_GRIPPER_NAME, Topics
from omx_pi05.gripper import norm_to_metres
from omx_pi05.kinematics import forward_kinematics

TASK = "pick up the cube and place it in the bowl"
CAM_H, CAM_W = 480, 640
LAG = 0.1  # s, follower behind the commanded target
T0 = 1_800_000_000.0  # s, start of episode 0 (ROS time)

TS = get_typestore(Stores.ROS2_JAZZY)
JointState = TS.types["sensor_msgs/msg/JointState"]
Image = TS.types["sensor_msgs/msg/Image"]
Header = TS.types["std_msgs/msg/Header"]
Time = TS.types["builtin_interfaces/msg/Time"]


def header(t: float, frame: str = "") -> Header:
    return Header(stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)) % 1_000_000_000), frame_id=frame)


def plan(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """Keyframe times (s from start), keyframe joints (K, 4) and gripper per keyframe, plus total duration."""
    home = np.array(C.HOME)
    pick = home + rng.uniform([-0.5, 0.6, -0.5, 0.2], [0.5, 0.9, -0.3, 0.5])
    place = home + rng.uniform([-0.8, 0.5, -0.5, 0.2], [0.8, 0.8, -0.3, 0.5])
    keys = [(0.0, home, 0.0), (1.0, home, 0.0), (3.5, pick, 0.0), (4.3, pick, 1.0), (6.8, place, 1.0),
            (7.6, place, 0.0), (10.1, home, 0.0), (11.1, home, 0.0)]
    times = np.array([k[0] for k in keys])
    return times, np.stack([k[1] for k in keys]), np.array([k[2] for k in keys])


def commanded(t: np.ndarray, times: np.ndarray, joints: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """(N, 5) commanded [j1..j4, gripper] at times t: min-jerk between keyframes, gripper ramps over each segment."""
    i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    s = np.clip((t - times[i]) / (times[i + 1] - times[i]), 0.0, 1.0)[:, None]
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    q = joints[i] + (joints[i + 1] - joints[i]) * blend
    g = grip[i] + (grip[i + 1] - grip[i]) * blend[:, 0]
    return np.column_stack([q, g])


def render_front(q: np.ndarray) -> np.ndarray:
    img = np.full((CAM_H, CAM_W, 3), (150, 120, 90), dtype=np.uint8)
    x, y, z, _ = forward_kinematics(q[:4])
    r, c, half = int(CAM_H - x / 0.4 * CAM_H), int(CAM_W / 2 - y / 0.53 * CAM_W), int(10 + 60 * max(z, 0.0))
    img[max(r - half, 0): max(r + half, 0), max(c - half, 0): max(c + half, 0)] = (30, 60, 200)
    return img


def render_wrist(q: np.ndarray) -> np.ndarray:
    img = np.full((CAM_H, CAM_W, 3), (110, 110, 110), dtype=np.uint8)
    gap = int(40 + 200 * (1 - q[4]))
    for c in (CAM_W // 2 - gap, CAM_W // 2 + gap):
        img[CAM_H - 80:CAM_H - 20, max(c - 25, 0):c + 25] = (60, 60, 60)
    return img


def joint_state(t: float, names: list[str], values: np.ndarray) -> JointState:
    n = len(names)
    return JointState(header=header(t), name=names, position=np.asarray(values, dtype=np.float64),
                      velocity=np.zeros(n), effort=np.zeros(n))


def image(t: float, rgb: np.ndarray, frame: str) -> Image:
    return Image(header=header(t, frame), height=CAM_H, width=CAM_W, encoding="rgb8", is_bigendian=0, step=CAM_W * 3,
                 data=rgb.reshape(-1))


def write_episode(path: pathlib.Path, start: float, rng: np.random.Generator, gap: bool, wrist: bool) -> None:
    times, joints, grip = plan(rng)
    duration = times[-1]
    topics = Topics()

    def follower(t: float) -> np.ndarray:
        """Measured [j1..j4 rad, gripper 0..1]: the command LAG seconds earlier, plus encoder noise."""
        q = commanded(np.array([t - start - LAG]), times, joints, grip)[0]
        q[:4] += rng.normal(0, 0.001, 4)
        return q

    streams = [  # (topic, msgtype, rate, transport delay, build(t) -> msg)
        (topics.action, JointState, 30.0, 0.002,
         lambda t: joint_state(t, [*C.ARM_JOINTS, ACTION_GRIPPER_NAME], commanded(np.array([t - start]), times, joints, grip)[0])),
        (topics.state, JointState, 100.0, 0.001,
         lambda t: joint_state(t, [C.GRIPPER_JOINT, *C.ARM_JOINTS], measured_row(follower(t)))),
        (topics.front, Image, 15.0, 0.02, lambda t: image(t, render_front(follower(t)), "front_color_optical_frame")),
    ]
    if wrist:
        streams.append((topics.wrist, Image, 15.0, 0.02, lambda t: image(t, render_wrist(follower(t)), "wrist_camera")))

    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        records = []
        for topic, msgtype, rate, delay, build in streams:
            conn = writer.add_connection(topic, msgtype.__msgtype__, typestore=TS)
            phase = rng.uniform(0, 1 / rate)
            for t in start + phase + np.arange(int(duration * rate)) / rate:
                if gap and topic == topics.front and start + 5.0 <= t < start + 5.5:
                    continue
                records.append((t + delay, conn, msgtype.__msgtype__, build(t)))
        for receive, conn, msgtype, msg in sorted(records, key=lambda r: r[0]):
            writer.write(conn, int(receive * 1e9), TS.serialize_cdr(msg, msgtype))


def measured_row(q: np.ndarray) -> np.ndarray:
    """/joint_states positions in the order [gripper_left_joint (m), joint1..4 (rad)], like ros2_control's broadcaster."""
    return np.append(norm_to_metres(q[4]), q[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent.parent / "data" / "rosbags" / "synthetic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-episode", type=int, default=-1, help="episode index with a 0.5 s front-camera dropout")
    parser.add_argument("--fail-episode", type=int, default=-1, help="episode index marked success=false")
    parser.add_argument("--no-wrist", action="store_true", help="record no wrist camera topic")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"❌ {args.out} exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        path = args.out / f"episode_{ep:04d}"
        write_episode(path, T0 + 100.0 * ep, rng, gap=ep == args.gap_episode, wrist=not args.no_wrist)
        meta = {"task": TASK, "success": ep != args.fail_episode, "episode_id": ep, "synthetic": True}
        (path / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"   {path.name}: {'gap, ' if ep == args.gap_episode else ''}{'failed' if not meta['success'] else 'success'}")
    print(f"✅ Wrote {args.episodes} synthetic episodes to {args.out}")


if __name__ == "__main__":
    main()
