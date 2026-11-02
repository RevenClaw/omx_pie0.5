"""omx_pi05.dataset + scripts/convert_rosbag_to_lerobot.py on synthetic rosbag2 episodes (scripts/make_synthetic_bags.py).

Needs rosbags (pinned in scripts/setup_server.sh) and openpi-client; skips without them. Images are shrunk to 160×120 so the
bags stay small; the converter resizes them to 224×224 like real 640×480 frames.
"""

import dataclasses
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("rosbags")
pytest.importorskip("openpi_client")

from conftest import load_script  # noqa: E402
from omx_pi05 import constants as C  # noqa: E402
from omx_pi05 import dataset as D  # noqa: E402


@pytest.fixture(scope="module")
def bags(tmp_path_factory):
    synth = load_script("make_synthetic_bags")
    synth.CAM_H, synth.CAM_W = 120, 160
    root = tmp_path_factory.mktemp("bags")
    rng = np.random.default_rng(0)
    specs = {"ok": {}, "gap": {"gap": True}, "failed": {}, "no_wrist": {"wrist": False}}
    for i, (name, kw) in enumerate(specs.items()):
        path = root / name
        synth.write_episode(path, synth.T0 + 100.0 * i, rng, gap=kw.get("gap", False), wrist=kw.get("wrist", True))
        meta = {"task": synth.TASK, "success": name != "failed"}
        (path / "metadata.json").write_text(json.dumps(meta))
    return root, synth


def test_clean_episode_is_aligned_and_trimmed(bags):
    root, synth = bags
    ep = D.load_episode(root / "ok")
    n = len(ep.times)
    assert ep.state.shape == ep.action.shape == (n, C.STATE_DIM) and ep.state.dtype == np.float32
    assert ep.front.shape == ep.wrist.shape == (n, C.IMAGE_SIZE, C.IMAGE_SIZE, 3) and ep.front.dtype == np.uint8
    np.testing.assert_allclose(np.diff(ep.times), 1 / C.FPS, atol=1e-6)  # absolute ROS times: ~0.2 µs float64 resolution
    assert ep.task == synth.TASK and ep.success
    # 1 s of idle at each end is cut down to the 0.3 s margin.
    assert ep.report["trim"]["recorded_s"] - ep.report["trim"]["kept_s"] > 1.0
    assert 0.0 <= ep.state[:, 4].min() < 0.05 and 0.95 < ep.state[:, 4].max() <= 1.0  # gripper metres → 0..1
    # The follower is LAG (0.1 s ≈ 1.5 frames) behind the command: best match at 1 or 2 frames.
    errors = [np.abs(ep.state[k:, :4] - ep.action[: n - k, :4]).mean() for k in range(5)]
    assert int(np.argmin(errors)) in (1, 2)
    for key, rate in (("state", 100), ("action", 30), ("front", 15), ("wrist", 15)):
        assert abs(ep.report[key]["rate_hz"] - rate) < 1.0
        assert ep.report[key]["max_skew_s"] <= 0.5 / C.FPS + 0.03 if key != "action" else ep.report[key]["max_skew_s"] <= 1 / 30 + 0.01


def test_gap_is_rejected(bags):
    root, _ = bags
    with pytest.raises(D.EpisodeError, match=r"front has a 0\.[56]\d s gap"):
        D.load_episode(root / "gap")


def test_failure_is_flagged(bags):
    root, _ = bags
    assert D.load_episode(root / "failed").success is False


def test_missing_wrist_camera(bags):
    root, _ = bags
    with pytest.raises(D.EpisodeError, match="topics not in the bag: wrist"):
        D.load_episode(root / "no_wrist")
    ep = D.load_episode(root / "no_wrist", topics=D.Topics(wrist=""))
    assert not ep.wrist.any() and ep.front.any()


def test_task_is_required(bags, tmp_path):
    root, _ = bags
    no_meta = tmp_path / "no_meta"
    shutil.copytree(root / "ok", no_meta, ignore=shutil.ignore_patterns("metadata.json"))
    with pytest.raises(D.EpisodeError, match="no task"):
        D.load_episode(no_meta)
    assert D.load_episode(no_meta, task="given").task == "given"
    assert D.load_episode(root / "ok", task="override").task == "override"


def test_find_episodes(bags):
    root, _ = bags
    assert [p.name for p in D.find_episodes(root)] == ["failed", "gap", "no_wrist", "ok"]
    assert D.find_episodes(root / "ok") == [root / "ok"]


def test_index_helpers():
    stamps = np.array([0.0, 0.1, 0.2, 0.3])
    t = np.array([0.0, 0.04, 0.06, 0.19, 0.31])
    np.testing.assert_array_equal(D.nearest_index(stamps, t), [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(D.latest_index(stamps, t), [0, 0, 0, 1, 3])  # the command in effect, never a future one


def test_active_range():
    cfg = D.AlignConfig(trim_margin=0.2)  # 3 frames at 15 Hz
    action = np.zeros((40, 5))
    action[10:30, 0] = np.linspace(0.1, 1.0, 20)  # moves from frame 10; frame 29 is the first at the final pose
    action[30:, 0] = 1.0
    # Keeps the last idle frame before the movement (9) and the first frame at rest (29), plus 3 margin frames each side.
    assert D.active_range(action, cfg) == (9 - 3, 29 + 3)
    with pytest.raises(D.EpisodeError, match="never changes"):
        D.active_range(np.zeros((40, 5)), cfg)


def test_features_match_data_config_keys():
    assert set(D.lerobot_features()) == {C.DS_FRONT, C.DS_WRIST, C.DS_STATE, C.DS_ACTION}


def test_converter_dry_run(bags, capsys):
    root, _ = bags
    conv = load_script("convert_rosbag_to_lerobot")  # skips without tyro (i.e. outside the openpi env)
    conv.main(conv.Args(input_dir=root, dry_run=True))
    out = capsys.readouterr().out
    assert "✅ ok:" in out and "skipped, front has a" in out and "skipped, marked success=false" in out
    assert "topics not in the bag: wrist" in out  # no_wrist without --no-wrist
    conv.main(dataclasses.replace(conv.Args(input_dir=root, dry_run=True), no_wrist=True))
    assert "✅ no_wrist:" in capsys.readouterr().out
