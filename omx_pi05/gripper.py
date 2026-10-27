"""Gripper convention (plan D6): metres on the robot ⇄ normalized [0, 1] in the model. 0 = open, 1 = closed."""

import numpy as np

from omx_pi05.constants import GRIPPER_CLOSE_M, GRIPPER_OPEN_M

_SPAN = GRIPPER_OPEN_M - GRIPPER_CLOSE_M  # 0.029 m


def metres_to_norm(q):
    """gripper_left_joint position (m) → [0, 1], clipped. 0.019 m (open) → 0, −0.010 m (closed) → 1."""
    return np.clip((GRIPPER_OPEN_M - np.asarray(q, dtype=np.float64)) / _SPAN, 0.0, 1.0)


def norm_to_metres(g):
    """[0, 1] → gripper_left_joint position (m), input clipped to [0, 1] first."""
    return GRIPPER_OPEN_M - np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0) * _SPAN
