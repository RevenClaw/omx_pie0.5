"""OpenMANIPULATOR-X facts used by the pi0.5 pipeline. Pure Python + numpy, no openpi/ROS imports.

Every robot value is copied from `low_level_controller` (paths relative to `low_level_controller/open_manipulator/`).
`tests/test_constants_match_robot.py` re-reads those files and fails if they drift (that repo duplicates values, issue #5).
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Joints and controllers
# (open_manipulator_bringup/config/open_manipulator_x/hardware_controller_manager.yaml)
# ---------------------------------------------------------------------------
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4")  # arm_controller, JointTrajectoryController, position
GRIPPER_JOINT = "gripper_left_joint"  # gripper_controller, GripperActionController (action only, metres)
CONTROLLER_RATE_HZ = 100  # controller_manager update_rate

# ---------------------------------------------------------------------------
# Joint limits, rad (open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro,
# same values as custom_package/joint_commander.py JOINT_LIMITS). j2/j3 were widened in known issue #2.
# ---------------------------------------------------------------------------
JOINT_LIMITS = {
    "joint1": (-math.pi, math.pi),
    "joint2": (-1.82, 1.5),
    "joint3": (-1.5, 1.55),
    "joint4": (-1.7, 1.97),
}
URDF_MAX_JOINT_VELOCITY = 4.8  # rad/s, URDF <limit velocity>. Physical arm overloads far below this (see below).

# Servo EEPROM position limits, ticks (ros2_control/open_manipulator_x_position.ros2_control.xacro, gpio dxl1–dxl4).
# rad = ticks * 2π/4096 − π. The URDF limits above lie inside these, so the URDF limits are the binding ones.
SERVO_POSITION_LIMITS_TICKS = {
    "joint1": (0, 4095),
    "joint2": (862, 3506),  # ≈ [−1.819, 2.237] rad
    "joint3": (712, 3094),  # ≈ [−2.049, 1.605] rad
    "joint4": (715, 3340),  # ≈ [−2.045, 1.982] rad
}


def ticks_to_rad(ticks: float) -> float:
    return ticks * 2 * math.pi / 4096 - math.pi


# ---------------------------------------------------------------------------
# Gripper (custom_package/joint_commander.py; URDF prismatic limit is [−0.011, 0.020] m)
# ---------------------------------------------------------------------------
GRIPPER_OPEN_M = 0.019  # commanded "open"
GRIPPER_CLOSE_M = -0.010  # commanded "close" (= ros2_control initial_value)
GRIPPER_URDF_LIMITS_M = (-0.011, 0.020)

# ---------------------------------------------------------------------------
# Named poses, rad [j1, j2, j3, j4]
# HOME: open_manipulator_bringup/config/open_manipulator_x/initial_positions.yaml (duplicated ~8×, issue #5)
# EXIT: custom_package/interactive_homing_node.py min_stress_exit_position = measured torque-off resting pose.
#       j2 there (−1.819) sits at the j2 lower limit; with torque off the arm rests at −1.839, *outside* the limit.
# ---------------------------------------------------------------------------
HOME = (0.6090, -1.3622, 1.3898, -0.0154)
MIN_STRESS_EXIT = (0.4633, -1.8190, 1.4972, 0.4019)

# ---------------------------------------------------------------------------
# Kinematics, m (custom_package/ee_commander.py, from the URDF; world = link1, x forward, y left, z up)
# ---------------------------------------------------------------------------
JOINT1_X = 0.012
JOINT2_Z = 0.0595
UPPER_ARM = (0.024, 0.128)  # joint2 → joint3 (x, z) at joint2 = 0
FOREARM = 0.124  # joint3 → joint4
GRIPPER_LEN = 0.126  # joint4 → end_effector_link (between the fingertips)

# ---------------------------------------------------------------------------
# Pipeline decisions (docs/IMPLEMENTATION_PLAN.md D4–D10)
# ---------------------------------------------------------------------------
FPS = 15  # record + control rate
ACTION_HORIZON = 10  # chunk length the model predicts
EXECUTE_STEPS = 5  # steps executed per chunk before re-querying
STATE_DIM = ACTION_DIM = 5  # [j1, j2, j3, j4, gripper_norm]
IMAGE_SIZE = 224

# Keys sent by the robot client / produced by the dataset repack (the policy-side contract)
OBS_FRONT = "observation/front_image"
OBS_WRIST = "observation/wrist_image"
OBS_STATE = "observation/state"
PROMPT = "prompt"

# LeRobot dataset feature names (what the converter writes)
DS_FRONT = "observation.images.front"
DS_WRIST = "observation.images.wrist"
DS_STATE = "observation.state"
DS_ACTION = "action"

# ---------------------------------------------------------------------------
# Safety constraints for anything that commands the arm (vla_bridge, Phase 3), from low_level_controller docs:
#  * joint2 (shoulder, XM430-W350) trips Overload 0x20 when moving fast in extended poses, and the arm then collapses.
#    Keep velocities low, lift the elbow (joint3) first (SYSTEM_WORKFLOW_ARCHITECTURE_MANUAL.md, "Joint 2 Overload").
#  * Never call /dynamixel_hardware_interface/get_dxl_data on a running arm: it stalls the 100 Hz loop (issue #8).
#  * Launch with init_position:=false for external control.
#  * Option 1 (hand pose) can leave joints outside JOINT_LIMITS, so the first observed state may be out of range.
# ---------------------------------------------------------------------------
JOINT_LIMIT_MARGIN = 0.05  # rad kept clear of JOINT_LIMITS when commanding
START_MAX_JOINT_VEL = 0.3  # rad/s cap for first hardware runs (plan T3.5, "25 %" stage)


def joint_limits_array(margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) arrays for joint1..joint4, shrunk by `margin` on each side."""
    lo = np.array([JOINT_LIMITS[j][0] for j in ARM_JOINTS]) + margin
    hi = np.array([JOINT_LIMITS[j][1] for j in ARM_JOINTS]) - margin
    return lo, hi
