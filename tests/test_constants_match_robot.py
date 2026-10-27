"""The robot facts in omx_pi05/constants.py must match low_level_controller (it duplicates values, issue #5)."""

import math
import re

import pytest

from conftest import LOW_LEVEL
from omx_pi05 import constants as C
from omx_pi05.kinematics import forward_kinematics

OM = LOW_LEVEL / "open_manipulator"
pytestmark = pytest.mark.skipif(not OM.exists(), reason="low_level_controller checkout not next to vla_layer")


def read(rel: str) -> str:
    return (OM / rel).read_text()


def test_urdf_joint_limits():
    xacro = read("open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro")
    for joint, (lo, hi) in C.JOINT_LIMITS.items():
        block = re.search(rf'name="\$\{{prefix\}}{joint}".*?<limit ([^/]*)/>', xacro, re.S).group(1)
        lower = re.search(r'lower="\$\{(.*?)\}"', block).group(1)
        upper = re.search(r'upper="\$\{(.*?)\}"', block).group(1)
        assert eval(lower, {"pi": math.pi}) == pytest.approx(lo), joint  # noqa: S307 - trusted local file
        assert eval(upper, {"pi": math.pi}) == pytest.approx(hi), joint  # noqa: S307


def test_home_pose():
    yaml = read("open_manipulator_bringup/config/open_manipulator_x/initial_positions.yaml")
    home = [float(v) for v in re.search(r"home:\s*\[(.*?)\]", yaml).group(1).split(",")]
    assert tuple(home) == C.HOME


def test_controllers():
    yaml = read("open_manipulator_bringup/config/open_manipulator_x/hardware_controller_manager.yaml")
    assert f"update_rate: {C.CONTROLLER_RATE_HZ}" in yaml
    assert all(f"- {j}" in yaml for j in C.ARM_JOINTS)
    assert f"joint: {C.GRIPPER_JOINT}" in yaml


def test_gripper_and_exit_pose():
    jc = read("custom_package/custom_package/joint_commander.py")
    assert float(re.search(r"GRIPPER_OPEN_M\s*=\s*([-\d.]+)", jc).group(1)) == C.GRIPPER_OPEN_M
    assert float(re.search(r"GRIPPER_CLOSE_M\s*=\s*([-\d.]+)", jc).group(1)) == C.GRIPPER_CLOSE_M
    homing = read("custom_package/custom_package/interactive_homing_node.py")
    exit_pose = re.search(r"min_stress_exit_position\s*=\s*\[(.*?)\]", homing).group(1)
    assert tuple(float(v) for v in exit_pose.split(",")) == C.MIN_STRESS_EXIT


def test_servo_limits():
    xacro = read("open_manipulator_description/ros2_control/open_manipulator_x_position.ros2_control.xacro")
    mins = [int(v) for v in re.findall(r'"Min Position Limit">(\d+)<', xacro)]
    maxs = [int(v) for v in re.findall(r'"Max Position Limit">(\d+)<', xacro)]
    assert list(zip(mins, maxs)) == list(C.SERVO_POSITION_LIMITS_TICKS.values())
    for joint, (lo, hi) in C.JOINT_LIMITS.items():  # URDF limits must lie inside the servo limits (± 1 tick)
        tlo, thi = C.SERVO_POSITION_LIMITS_TICKS[joint]
        assert C.ticks_to_rad(tlo) <= lo + 2e-3 and hi <= C.ticks_to_rad(thi) + 2e-3, joint


def test_kinematics_constants_match_ee_commander():
    ee = read("custom_package/custom_package/ee_commander.py")
    for name, value in [("JOINT1_X", C.JOINT1_X), ("JOINT2_Z", C.JOINT2_Z), ("FOREARM", C.FOREARM), ("GRIPPER", C.GRIPPER_LEN)]:
        assert float(re.search(rf"^{name}\s*=\s*([\d.]+)", ee, re.M).group(1)) == value, name
    assert re.search(r"^UPPER_ARM\s*=\s*\(([\d.]+),\s*([\d.]+)\)", ee, re.M).groups() == tuple(map(str, C.UPPER_ARM))


def test_home_is_above_table():
    assert forward_kinematics(C.HOME)[2] > 0.05
