import numpy as np
import pytest

from omx_pi05.constants import GRIPPER_CLOSE_M, GRIPPER_OPEN_M
from omx_pi05.gripper import metres_to_norm, norm_to_metres


def test_endpoints():
    assert metres_to_norm(GRIPPER_OPEN_M) == 0.0  # open → 0
    assert metres_to_norm(GRIPPER_CLOSE_M) == 1.0  # closed → 1
    assert norm_to_metres(0.0) == pytest.approx(GRIPPER_OPEN_M)
    assert norm_to_metres(1.0) == pytest.approx(GRIPPER_CLOSE_M)


def test_clipping():
    assert metres_to_norm(0.020) == 0.0  # URDF upper limit, wider than "open"
    assert metres_to_norm(-0.011) == 1.0  # URDF lower limit
    assert norm_to_metres(1.5) == pytest.approx(GRIPPER_CLOSE_M)
    assert norm_to_metres(-0.5) == pytest.approx(GRIPPER_OPEN_M)


def test_round_trip_and_monotonic():
    q = np.linspace(GRIPPER_CLOSE_M, GRIPPER_OPEN_M, 50)
    np.testing.assert_allclose(norm_to_metres(metres_to_norm(q)), q, atol=1e-12)
    assert np.all(np.diff(metres_to_norm(q)) < 0)  # opening (larger q) → smaller value
