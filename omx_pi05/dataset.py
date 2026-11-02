"""Robot recordings (rosbag2) → aligned 15 Hz episodes → LeRobot frames (plan Phase 5, T5.1/T5.2). No ROS needed.

One episode = one rosbag2 directory (mcap or sqlite3), as written by low_level_controller's episode recorder, with a
`metadata.json` next to the bag's `metadata.yaml`: {"task": "<prompt>", "success": true, ...}.

Streams read from the bag (topic names: `Topics`):
  state   /joint_states                  sensor_msgs/JointState  joint1..4 (rad) + gripper_left_joint (m) → gripper 0 open..1 closed
  action  /teleop/action                 sensor_msgs/JointState  joint1..4 (rad) + "gripper" (already 0 open..1 closed): the
                                                                 target the teleop node COMMANDED (plan T4.4), not the measured pose
  front   /camera/front/color/image_raw  sensor_msgs/Image       rgb8/bgr8 → RGB → resize_with_pad 224×224 (plan D8)
  wrist   /camera/wrist/image_raw        sensor_msgs/Image       same; topic "" = no wrist camera → black frames

Alignment onto a 15 Hz grid (header stamps; the bag's receive time only if a stamp is zero):
  - the grid starts when every stream has started and ends when the first stream ends;
  - state and images: the message NEAREST to each grid time (a 15 Hz camera is then at most ~33 ms off);
  - action: the LATEST message at or before each grid time, i.e. the command in effect at that moment;
Idle frames at both ends (operator not yet moving / already done) are trimmed by watching the action, keeping a margin.
Then the episode is rejected if any stream has a gap longer than `max_gap` inside the KEPT part (dropped frames, a stalled
node). Gaps in the trimmed idle part don't matter: e.g. rosbag2 often misses a few messages right after it starts.
"""

import dataclasses
import json
import pathlib

import numpy as np
from openpi_client import image_tools

from omx_pi05 import constants as C
from omx_pi05.gripper import metres_to_norm

ACTION_GRIPPER_NAME = "gripper"  # name of the normalized gripper entry in /teleop/action


@dataclasses.dataclass(frozen=True)
class Topics:
    state: str = "/joint_states"
    action: str = "/teleop/action"
    front: str = "/camera/front/color/image_raw"
    wrist: str = "/camera/wrist/image_raw"  # "" = no wrist camera: black frames


@dataclasses.dataclass(frozen=True)
class AlignConfig:
    fps: int = C.FPS
    max_gap: float = 0.2  # s; a longer gap in any stream rejects the episode
    trim: bool = True
    trim_margin: float = 0.3  # s of idle kept before the first and after the last movement
    joint_tol: float = 0.01  # rad; action change that counts as movement
    gripper_tol: float = 0.05  # normalized
    min_duration: float = 2.0  # s after trimming


@dataclasses.dataclass
class Stream:
    stamps: np.ndarray  # (N,) seconds, sorted
    values: np.ndarray  # (N, ...) one row per message


@dataclasses.dataclass
class Episode:
    name: str
    task: str
    success: bool
    times: np.ndarray  # (T,) grid times, seconds
    state: np.ndarray  # (T, 5) float32
    action: np.ndarray  # (T, 5) float32
    front: np.ndarray  # (T, 224, 224, 3) uint8
    wrist: np.ndarray  # (T, 224, 224, 3) uint8
    report: dict  # rates, gaps, skews, trimming (JSON-serializable)


class EpisodeError(ValueError):
    """The episode can't be converted (missing stream, gap, no motion, too short, ...)."""


def lerobot_features() -> dict:
    """LeRobot features of an OM-X dataset (keys = omx_pi05.data_config's repack inputs)."""
    image = {"dtype": "image", "shape": (C.IMAGE_SIZE, C.IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    vec = {"dtype": "float32", "shape": (C.STATE_DIM,), "names": [*C.ARM_JOINTS, "gripper"]}
    return {C.DS_FRONT: image, C.DS_WRIST: dict(image), C.DS_STATE: vec, C.DS_ACTION: dict(vec)}


def find_episodes(root: pathlib.Path) -> list[pathlib.Path]:
    """Episode directories under `root` (any depth): every directory holding a rosbag2 metadata.yaml, sorted by path."""
    root = pathlib.Path(root)
    if (root / "metadata.yaml").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("metadata.yaml"))


def read_metadata(episode_dir: pathlib.Path) -> dict:
    path = pathlib.Path(episode_dir) / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------------------------------------- reading
def image_to_rgb(encoding: str, height: int, width: int, step: int, data) -> np.ndarray:
    """sensor_msgs/Image fields (rgb8 or bgr8) → uint8 (H, W, 3) RGB. Same logic as the bridge's vla_safety.ros_image_to_rgb."""
    if encoding not in ("rgb8", "bgr8"):
        raise EpisodeError(f"unsupported image encoding {encoding!r} (record rgb8 or bgr8)")
    rows = np.frombuffer(memoryview(data), dtype=np.uint8).reshape(height, step)
    image = rows[:, : width * 3].reshape(height, width, 3)
    return np.ascontiguousarray(image[..., ::-1] if encoding == "bgr8" else image)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Exactly what vla_bridge sends the model at run time (plan D8)."""
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, C.IMAGE_SIZE, C.IMAGE_SIZE))


def _stamp(msg, receive_ns: int) -> float:
    s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return s if s > 0 else receive_ns * 1e-9


def _joint_row(msg, gripper_name: str, gripper_in_metres: bool):
    names = list(msg.name)
    if not all(n in names for n in (*C.ARM_JOINTS, gripper_name)):
        return None
    row = [msg.position[names.index(j)] for j in C.ARM_JOINTS]
    g = msg.position[names.index(gripper_name)]
    return [*row, float(metres_to_norm(g)) if gripper_in_metres else float(np.clip(g, 0.0, 1.0))]


def read_bag(bag_dir: pathlib.Path, topics: Topics = Topics()) -> dict[str, Stream]:
    """Reads the episode's streams. Images are preprocessed to 224×224 while reading, so memory stays small."""
    from rosbags.highlevel import AnyReader  # imported here so the rest of omx_pi05 works without rosbags
    from rosbags.typesys import Stores, get_typestore

    wanted = {"state": topics.state, "action": topics.action, "front": topics.front}
    if topics.wrist:
        wanted["wrist"] = topics.wrist
    rows: dict[str, list] = {k: [] for k in wanted}
    with AnyReader([pathlib.Path(bag_dir)], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        present = {c.topic for c in reader.connections}
        missing = [f"{k} ({t})" for k, t in wanted.items() if t not in present]
        if missing:
            raise EpisodeError(f"topics not in the bag: {', '.join(missing)}")
        by_topic = {t: k for k, t in wanted.items()}
        connections = [c for c in reader.connections if c.topic in by_topic]
        for conn, receive_ns, raw in reader.messages(connections=connections):
            key = by_topic[conn.topic]
            msg = reader.deserialize(raw, conn.msgtype)
            if key == "state":
                value = _joint_row(msg, C.GRIPPER_JOINT, gripper_in_metres=True)
            elif key == "action":
                value = _joint_row(msg, ACTION_GRIPPER_NAME, gripper_in_metres=False)
            else:
                value = preprocess(image_to_rgb(msg.encoding, msg.height, msg.width, msg.step, msg.data))
            if value is not None:
                rows[key].append((_stamp(msg, receive_ns), value))

    streams = {}
    for key, items in rows.items():
        if not items:
            raise EpisodeError(f"no usable {key} messages on {wanted[key]}")
        items.sort(key=lambda r: r[0])
        streams[key] = Stream(np.array([r[0] for r in items]), np.stack([np.asarray(r[1]) for r in items]))
    return streams


# --------------------------------------------------------------------------------------------------------- aligning
def nearest_index(stamps: np.ndarray, t: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1)
    return np.where(np.abs(t - stamps[i - 1]) <= np.abs(stamps[i] - t), i - 1, i)


def latest_index(stamps: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(stamps, t, side="right") - 1, 0, len(stamps) - 1)


def active_range(action: np.ndarray, cfg: AlignConfig) -> tuple[int, int]:
    """First and last frame (inclusive) of the movement, plus the margin. Raises if the action never changes."""

    def differs(ref):
        return (np.abs(action[:, :4] - ref[:4]) > cfg.joint_tol).any(axis=1) | (np.abs(action[:, 4] - ref[4]) > cfg.gripper_tol)

    after_start, before_end = np.flatnonzero(differs(action[0])), np.flatnonzero(differs(action[-1]))
    if len(after_start) == 0:
        raise EpisodeError("the action never changes (no movement recorded)")
    margin = int(round(cfg.trim_margin * cfg.fps))
    return max(after_start[0] - 1 - margin, 0), min(before_end[-1] + 1 + margin, len(action) - 1)


def align(streams: dict[str, Stream], fps: int = C.FPS) -> tuple[np.ndarray, dict]:
    """Returns (grid times, {stream: row index per grid time})."""
    start = max(s.stamps[0] for s in streams.values())
    end = min(s.stamps[-1] for s in streams.values())
    if end - start < 1.0 / fps:
        raise EpisodeError("the streams don't overlap in time")
    grid = start + np.arange(int(np.floor((end - start) * fps)) + 1) / fps
    index = {k: latest_index(s.stamps, grid) if k == "action" else nearest_index(s.stamps, grid) for k, s in streams.items()}
    return grid, index


def stream_report(streams: dict[str, Stream], grid: np.ndarray, index: dict, cfg: AlignConfig) -> tuple[dict, list[str]]:
    """Per-stream rate, largest gap and largest time skew over the grid span, and the gaps longer than cfg.max_gap."""
    t0, t1 = grid[0] - 0.5 / cfg.fps, grid[-1] + 0.5 / cfg.fps
    report, problems = {}, []
    for key, s in streams.items():
        first, last = latest_index(s.stamps, np.array([t0]))[0], nearest_index(s.stamps, np.array([t1]))[0]
        span = s.stamps[first:max(last, first) + 1]  # messages covering [t0, t1], including the one held at t0
        max_gap = float(np.max(np.diff(span))) if len(span) > 1 else float("inf")
        report[key] = {
            "rate_hz": round(float((len(span) - 1) / max(span[-1] - span[0], 1e-9)), 1) if len(span) > 1 else 0.0,
            "max_gap_s": round(max_gap, 3),
            "max_skew_s": round(float(np.max(np.abs(grid - s.stamps[index[key]]))), 3),
        }
        if max_gap > cfg.max_gap:
            problems.append(f"{key} has a {max_gap:.2f} s gap (max {cfg.max_gap} s)")
    return report, problems


def load_episode(episode_dir: pathlib.Path, topics: Topics = Topics(), cfg: AlignConfig = AlignConfig(),
                 task: str | None = None) -> Episode:
    """Reads, aligns and trims one episode. `task` overrides metadata.json's task."""
    episode_dir = pathlib.Path(episode_dir)
    meta = read_metadata(episode_dir)
    task = task or meta.get("task")
    if not task:
        raise EpisodeError("no task: metadata.json has no 'task' and none was given")
    streams = read_bag(episode_dir, topics)
    grid, idx = align(streams, cfg.fps)

    first, last = active_range(streams["action"].values[idx["action"]], cfg) if cfg.trim else (0, len(grid) - 1)
    sl = slice(first, last + 1)
    idx = {k: v[sl] for k, v in idx.items()}
    report, problems = stream_report(streams, grid[sl], idx, cfg)
    duration = (last - first) / cfg.fps
    report["trim"] = {"recorded_s": round((len(grid) - 1) / cfg.fps, 2), "kept_s": round(duration, 2),
                      "first_frame": int(first), "last_frame": int(last)}
    if problems:
        raise EpisodeError("; ".join(problems))
    if duration < cfg.min_duration:
        raise EpisodeError(f"only {duration:.1f} s left after trimming (min {cfg.min_duration} s)")

    n = last - first + 1
    black = np.zeros((n, C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8)
    return Episode(
        name=episode_dir.name,
        task=task,
        success=bool(meta.get("success", True)),
        times=grid[sl],
        state=streams["state"].values[idx["state"]].astype(np.float32),
        action=streams["action"].values[idx["action"]].astype(np.float32),
        front=streams["front"].values[idx["front"]],
        wrist=streams["wrist"].values[idx["wrist"]] if "wrist" in streams else black,
        report=report,
    )
