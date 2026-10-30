"""The robot-side vla_bridge must send observations exactly as omx_pi05 (training + serving) expects them.

Loads low_level_controller's vla_safety.py by path (it is pure numpy, no ROS). Skips where that repo isn't checked out next to
vla_layer (e.g. on the A30 server), like test_constants_match_robot.py.
"""

import importlib.util

import numpy as np
import pytest

from conftest import LOW_LEVEL
from omx_pi05 import constants as C
from omx_pi05.gripper import metres_to_norm, norm_to_metres

SAFETY = LOW_LEVEL / "open_manipulator" / "custom_package" / "custom_package" / "vla_safety.py"
pytestmark = pytest.mark.skipif(not SAFETY.exists(), reason="low_level_controller checkout not next to vla_layer")


@pytest.fixture(scope="module")
def bridge():
    spec = importlib.util.spec_from_file_location("vla_safety", SAFETY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_keys_and_sizes(bridge):
    assert (bridge.OBS_FRONT, bridge.OBS_WRIST, bridge.OBS_STATE, bridge.PROMPT) == (
        C.OBS_FRONT, C.OBS_WRIST, C.OBS_STATE, C.PROMPT)
    assert bridge.IMAGE_SIZE == C.IMAGE_SIZE and bridge.ACTION_DIM == C.ACTION_DIM


def test_gripper_formula(bridge):
    assert (bridge.GRIPPER_OPEN_M, bridge.GRIPPER_CLOSE_M) == (C.GRIPPER_OPEN_M, C.GRIPPER_CLOSE_M)
    for q in np.linspace(-0.012, 0.021, 23):
        assert bridge.gripper_metres_to_norm(q) == pytest.approx(float(metres_to_norm(q)))
    for g in np.linspace(-0.2, 1.2, 15):
        assert bridge.gripper_norm_to_metres(g) == pytest.approx(float(norm_to_metres(g)))
