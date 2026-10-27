import numpy as np

from conftest import load_script
from omx_pi05 import constants as C


def test_fake_motions_respect_robot_limits():
    synth = load_script("make_synthetic_dataset")  # skips without lerobot (i.e. outside the openpi env)
    lo, hi = C.joint_limits_array()
    rng = np.random.default_rng(1)
    for _ in range(20):
        joints, grip, grasped = synth.make_episode(rng)
        assert np.all(joints >= lo) and np.all(joints <= hi)
        assert np.max(np.abs(np.diff(joints, axis=0))) * C.FPS <= synth.MAX_VEL + 1e-6
        np.testing.assert_allclose(joints[0], C.HOME)
        np.testing.assert_allclose(joints[-1], C.HOME)
        assert set(np.unique(grip)) == {0.0, 1.0} and grasped.any()


def test_rendered_images_are_224():
    synth = load_script("make_synthetic_dataset")
    front, wrist = synth.render(np.array(C.HOME), (0.2, 0.0), 0.5)
    assert front.shape == wrist.shape == (224, 224, 3) and front.dtype == np.uint8
