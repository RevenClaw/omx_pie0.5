"""Phase 5 (T5.1/T5.2/T5.4): convert recorded OM-X episodes (rosbag2 + metadata.json) into a LeRobot dataset for training.

    source scripts/env.sh
    python scripts/convert_rosbag_to_lerobot.py data/rosbags/pick_cube --repo-id local/omx_pick_cube --dry-run   # check only
    python scripts/convert_rosbag_to_lerobot.py data/rosbags/pick_cube --repo-id local/omx_pick_cube              # write
    python scripts/convert_rosbag_to_lerobot.py data/rosbags/pick_cube --repo-id local/omx_pick_cube --overwrite --no-wrist

Input: a folder of episode directories (any depth), each a rosbag2 directory with a metadata.json from the recorder.
Each episode is read, aligned onto the 15 Hz grid, idle-trimmed and checked (see omx_pi05/dataset.py for exactly how).
Episodes with problems, or marked success=false (unless --include-failures), are skipped with the reason printed.
Output: $HF_LEROBOT_HOME/<repo-id> (openpi's pinned LeRobot format, same features as make_synthetic_dataset.py), plus
conversion_report.json in it: per-episode stats, skipped episodes and why, the dataset hash, and the git commit.
Then: add the repo id to omx_pi05/train_configs.py DATASETS, compute norm stats, train (docs/PHASE5_GUIDE.md).
"""

import dataclasses
import hashlib
import json
import pathlib
import shutil
import subprocess
import time
from collections import Counter

import tyro

from omx_pi05 import constants as C
from omx_pi05 import dataset as D

REPO = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Args:
    input_dir: tyro.conf.Positional[pathlib.Path]
    """Folder holding the recorded episode directories."""
    repo_id: str = ""
    """LeRobot dataset id under $HF_LEROBOT_HOME, e.g. local/omx_pick_cube. Not needed with --dry-run."""
    task: str | None = None
    """Prompt for every episode (overrides metadata.json). Must be exactly the string used at run time."""
    include_failures: bool = False
    """Also convert episodes the operator marked success=false."""
    no_wrist: bool = False
    """No wrist camera was recorded: write black wrist frames (the policy must then mask that view)."""
    dry_run: bool = False
    """Read and check every episode, print the report, write nothing."""
    overwrite: bool = False
    """Replace an existing dataset with the same repo id."""
    state_topic: str = D.Topics.state
    action_topic: str = D.Topics.action
    front_topic: str = D.Topics.front
    wrist_topic: str = D.Topics.wrist
    max_gap: float = D.AlignConfig.max_gap
    """Seconds; an episode with a longer gap in any stream (inside the kept part) is skipped."""
    trim: bool = True
    """Trim idle frames at both ends (--no-trim keeps the whole recording)."""
    trim_margin: float = D.AlignConfig.trim_margin
    min_duration: float = D.AlignConfig.min_duration


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_hash(episodes: list[D.Episode], commit: str, cfg: D.AlignConfig) -> str:
    """Identifies exactly which data went in (plan T5.4): episode names, frame counts, tasks, settings, converter commit."""
    h = hashlib.sha256()
    for e in episodes:
        h.update(f"{e.name}|{len(e.times)}|{e.task}|{e.times[0]:.3f}\n".encode())
    h.update(f"{commit}|{dataclasses.asdict(cfg)}".encode())
    return h.hexdigest()[:12]


def main(args: Args) -> None:
    topics = D.Topics(args.state_topic, args.action_topic, args.front_topic, "" if args.no_wrist else args.wrist_topic)
    cfg = D.AlignConfig(max_gap=args.max_gap, trim=args.trim, trim_margin=args.trim_margin, min_duration=args.min_duration)
    dirs = D.find_episodes(args.input_dir)
    if not dirs:
        raise SystemExit(f"❌ No rosbag2 episodes (metadata.yaml) under {args.input_dir}")
    if not args.dry_run and not args.repo_id:
        raise SystemExit("❌ --repo-id is required to write a dataset (or use --dry-run)")

    out = None
    if not args.dry_run:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset  # openpi's pinned LeRobot

        out = HF_LEROBOT_HOME / args.repo_id
        if out.exists():
            if not args.overwrite:
                raise SystemExit(f"❌ {out} exists. Pass --overwrite to replace it.")
            shutil.rmtree(out)
        dataset = LeRobotDataset.create(repo_id=args.repo_id, robot_type="openmanipulator_x", fps=C.FPS,
                                        features=D.lerobot_features(), image_writer_threads=8, image_writer_processes=2)

    print(f"📂 {len(dirs)} episode(s) in {args.input_dir}" + ("  (dry run: nothing is written)" if args.dry_run else f"  → {out}"))
    kept, skipped, started = [], [], time.time()
    for path in dirs:
        rel = str(path.relative_to(args.input_dir)) if path != args.input_dir else path.name
        try:
            ep = D.load_episode(path, topics, cfg, task=args.task)
        except D.EpisodeError as e:
            skipped.append({"episode": rel, "reason": str(e)})
            print(f"   ⚠️  {rel}: skipped, {e}")
            continue
        if not ep.success and not args.include_failures:
            skipped.append({"episode": rel, "reason": "marked success=false"})
            print(f"   ⏭️  {rel}: skipped, marked success=false")
            continue
        if not args.dry_run:
            for t in range(len(ep.times)):
                dataset.add_frame({C.DS_FRONT: ep.front[t], C.DS_WRIST: ep.wrist[t], C.DS_STATE: ep.state[t],
                                   C.DS_ACTION: ep.action[t], "task": ep.task})
            dataset.save_episode()
        kept.append(ep)
        rates = ", ".join(f"{k} {v['rate_hz']:g} Hz" for k, v in ep.report.items() if k != "trim")
        print(f"   ✅ {rel}: {len(ep.times)} frames ({ep.report['trim']['kept_s']:.1f} s of {ep.report['trim']['recorded_s']:.1f} s); {rates}")

    if not kept:
        raise SystemExit("❌ No episode could be converted (see the reasons above).")
    tasks = Counter(e.task for e in kept)
    frames = sum(len(e.times) for e in kept)
    commit = git_commit()
    report = {
        "repo_id": args.repo_id or None,
        "dataset_hash": dataset_hash(kept, commit, cfg),
        "converter_commit": commit,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(pathlib.Path(args.input_dir).resolve()),
        "settings": {**dataclasses.asdict(cfg), "topics": dataclasses.asdict(topics), "include_failures": args.include_failures},
        "episodes": len(kept),
        "frames": frames,
        "tasks": dict(tasks),
        "converted": [{"episode": e.name, "task": e.task, "frames": len(e.times), **e.report} for e in kept],
        "skipped": skipped,
    }
    if out is not None:
        (out / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'🔎 Dry run' if args.dry_run else '✅ Wrote'}: {len(kept)} episode(s), {frames} frames "
          f"({frames / C.FPS / 60:.1f} min at {C.FPS} Hz), {len(skipped)} skipped, in {time.time() - started:.0f} s")
    print(f"   dataset hash {report['dataset_hash']} (converter {commit}). Log it in docs/experiments.md.")
    for task, n in tasks.items():
        print(f"   task: \"{task}\" × {n}")
    if len(tasks) > 1:
        print("   ⚠️  More than one task string. Intended? Each must match the run-time prompt exactly (plan D12).")
    if args.no_wrist:
        print("   ⚠️  Wrist frames are black: the policy must mask the wrist view for training and serving (PHASE4_GUIDE Q2).")


if __name__ == "__main__":
    main(tyro.cli(Args))
