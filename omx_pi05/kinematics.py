"""Forward kinematics of the OM-X, same model as custom_package/ee_commander.py forward_kinematics()."""

import math

from omx_pi05.constants import FOREARM, GRIPPER_LEN, JOINT1_X, JOINT2_Z, UPPER_ARM

_UPPER_ARM_LEN = math.hypot(*UPPER_ARM)
_UPPER_ARM_ANGLE = math.atan2(UPPER_ARM[1], UPPER_ARM[0])


def forward_kinematics(joints) -> tuple[float, float, float, float]:
    """(x, y, z, pitch) of end_effector_link in world (= link1) for [j1, j2, j3, j4]. Pitch 0 = horizontal, +π/2 = down."""
    j1, j2, j3, j4 = (float(j) for j in joints[:4])
    phi_upper = j2 - _UPPER_ARM_ANGLE
    phi_fore = j2 + j3
    pitch = j2 + j3 + j4
    r = _UPPER_ARM_LEN * math.cos(phi_upper) + FOREARM * math.cos(phi_fore) + GRIPPER_LEN * math.cos(pitch)
    h = -(_UPPER_ARM_LEN * math.sin(phi_upper) + FOREARM * math.sin(phi_fore) + GRIPPER_LEN * math.sin(pitch))
    return (JOINT1_X + r * math.cos(j1), r * math.sin(j1), JOINT2_Z + h, pitch)
