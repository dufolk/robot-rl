"""站立与爬起奖励。

站立课按 Isaac Lab 的速度任务，命令恒为 0。水平速度和转向各用 exp 核对着 0，
再乘以躯干立起来的程度。躺平不动不再得这两项。

爬起课先只做发现：高度、高度增量、立起、双脚承重和脚力增量。
不扣平滑、力矩、速度和着地，避免挡住翻身。
getup-plus 在这五项上再叠站立课的一整套。
getup-human 和 impact 只用 DeepMimic 的相位跟踪模仿奖励。
固定局长只记评估用的成败，不加局末奖惩。

骨盆或胸口碰到地面在站立课用来结束这一局，爬起课不因此结束。小球打在身上不算落地。
"""

from __future__ import annotations

import math

import numpy as np

from robot_rl.robot import FALL_UPRIGHT

CHEST_UP_HEIGHT = 0.8
STABLE_LIN = 0.45
STABLE_ANG = 1.2

# Isaac Lab RewardsCfg / G1：命令为 0，std = 0.5。
TRACK_STD = 0.5
TRACK_LIN_VEL_XY = 1.0
TRACK_ANG_VEL_Z = 1.0
LIN_VEL_Z_L2 = -2.0
ANG_VEL_XY_L2 = -0.05
FLAT_ORIENTATION_L2 = -1.0
ACTION_RATE_L2 = -0.005
DOF_ACC_L2 = -1.25e-7
DOF_TORQUES_L2 = -1.5e-7
FEET_SLIDE = -0.1
UNDESIRED_CONTACTS = -1.0
# 行走任务用腾空时间鼓励迈步。这里不走路，改成拉向站立关节角。
JOINT_POSE_L1 = -0.1
# 着地已经会结束这一局。再扣 200 会把回报打成常数，价值网络学不出解释方差。
TERMINATION_PENALTY = 0.0
# 骨盆或胸口碰到地面的法向力超过这个值，单位 N，这一局结束。
TORSO_FORCE = 1.0

# 旧 getup-plus 的任务奖励，保留供旧模型使用。
GETUP_BASE_TARGET = 0.95
GETUP_HEIGHT = 2.0
GETUP_UPRIGHT = 2.0
GETUP_DHEIGHT = 1.0
GETUP_DFEET = 0.5
# 每只脚承担一成以上体重算撑住，两只都撑住时这项为 1.5。
FOOT_LOAD_FRACTION = 0.1
GETUP_FEET = 1.5
_FOOT_SENSORS = 2
_HEIGHT_RATE = 0.02
_FEET_RATE = 0.2


class StandReward:
    def __init__(self, weight: float) -> None:
        if weight <= 0.0:
            raise ValueError("体重应该是正的")
        self.weight = float(weight)
        self.reset()

    def reset(self) -> None:
        self._previous_action: np.ndarray | None = None

    def update(self, action: np.ndarray, info: dict) -> tuple[float, dict[str, float]]:
        command, rate = self._consume_action(action)
        terms = _stand_terms(command, rate, info, joint_pose=True)
        total = float(sum(terms.values()))
        terms["total"] = total
        return total, terms

    def _consume_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        previous = self._previous_action
        if previous is None or previous.shape != command.shape:
            previous = command
        rate = float(np.sum(np.square(command - previous)))
        self._previous_action = command.copy()
        return command, rate


class GetupReward(StandReward):
    """爬起发现：只奖升高、立起和双脚承重。"""

    def reset(self) -> None:
        super().reset()
        self._previous_height: float | None = None
        self._previous_feet: float | None = None

    def update(self, action: np.ndarray, info: dict) -> tuple[float, dict[str, float]]:
        gravity = np.asarray(info["projected_gravity"], dtype=np.float64)
        upright = max(0.0, float(-gravity[2]))
        base = max(0.0, float(info["base_height"]))
        previous = base if self._previous_height is None else self._previous_height
        floor = np.asarray(info["floor_n"], dtype=np.float64)
        feet_load = float(np.sum(floor[:_FOOT_SENSORS])) / self.weight
        last_feet = feet_load if self._previous_feet is None else self._previous_feet
        self._previous_height = base
        self._previous_feet = feet_load
        terms = {
            "height": GETUP_HEIGHT * min(base / GETUP_BASE_TARGET, 1.0),
            "d_height": GETUP_DHEIGHT * float(np.clip((base - previous) / _HEIGHT_RATE, -1.0, 1.0)),
            "upright": GETUP_UPRIGHT * upright,
            "feet": GETUP_FEET * _feet_support(info, self.weight),
            "d_feet": GETUP_DFEET * float(np.clip((feet_load - last_feet) / _FEET_RATE, -1.0, 1.0)),
        }
        total = float(sum(terms.values()))
        terms["total"] = total
        return total, terms


class GetupPlusReward(GetupReward):
    """发现奖励叠上站立课的一整套，用来压住滑步和抖动。"""

    def update(self, action: np.ndarray, info: dict) -> tuple[float, dict[str, float]]:
        _, getup_terms = super().update(action, info)
        command, rate = self._consume_action(action)
        terms = {key: value for key, value in getup_terms.items() if key != "total"}
        terms.update(_stand_terms(command, rate, info, joint_pose=True))
        total = float(sum(terms.values()))
        terms["total"] = total
        return total, terms


class DeepMimicReward(StandReward):
    """DeepMimic-style tracking reward; reference metadata is never an actor input."""

    def update(self, action: np.ndarray, info: dict) -> tuple[float, dict[str, float]]:
        from robot_rl.motion import match_reference

        motion = match_reference(
            np.asarray(info["joint_q"]), np.asarray(info["joint_velocity"]),
            np.asarray(info["motion_features"]),
            clip=info["reference_clip"], time=info["reference_time"],
        )
        value = motion["imitation"]
        info["motion_match"] = motion
        return value, {"imitation": value, "total": value}


def _stand_terms(command: np.ndarray, rate: float, info: dict, *, joint_pose: bool) -> dict[str, float]:
    linear = np.asarray(info["base_lin_vel"], dtype=np.float64)
    angular = np.asarray(info["base_ang_vel"], dtype=np.float64)
    gravity = np.asarray(info["projected_gravity"], dtype=np.float64)
    yaw = float(info["yaw_rate"])
    upright = max(0.0, float(-gravity[2]))
    variance = TRACK_STD * TRACK_STD
    floor = np.asarray(info["floor_n"], dtype=np.float64)
    contacts = int(np.sum(floor[_FOOT_SENSORS:] > TORSO_FORCE))
    torso = float(info["torso_floor_n"]) > TORSO_FORCE
    joint_pos = np.asarray(info["joint_pos"], dtype=np.float64)
    joint_acc = np.asarray(info["joint_acc"], dtype=np.float64)
    torque = np.asarray(info["torque"], dtype=np.float64)
    pose_pen = JOINT_POSE_L1 * float(np.sum(np.abs(joint_pos))) if joint_pose else 0.0
    return {
        "lin_vel": upright * TRACK_LIN_VEL_XY * math.exp(-float(linear[0] ** 2 + linear[1] ** 2) / variance),
        "ang_vel": upright * TRACK_ANG_VEL_Z * math.exp(-(yaw * yaw) / variance),
        "lin_vel_z": LIN_VEL_Z_L2 * float(linear[2] ** 2),
        "ang_vel_xy": ANG_VEL_XY_L2 * float(angular[0] ** 2 + angular[1] ** 2),
        "flat": FLAT_ORIENTATION_L2 * float(gravity[0] ** 2 + gravity[1] ** 2),
        "action_rate": ACTION_RATE_L2 * rate,
        "dof_acc": DOF_ACC_L2 * float(np.sum(np.square(joint_acc))),
        "dof_torque": DOF_TORQUES_L2 * float(np.sum(np.square(torque))),
        "feet_slide": FEET_SLIDE * float(info["foot_slip"]),
        "undesired": UNDESIRED_CONTACTS * float(contacts),
        "joint_pose": pose_pen,
        "termination": TERMINATION_PENALTY if torso or info.get("diverged") else 0.0,
    }


def _feet_support(info: dict, weight: float) -> float:
    floor = np.asarray(info["floor_n"], dtype=np.float64)
    loaded = floor[:_FOOT_SENSORS] > FOOT_LOAD_FRACTION * weight
    return float(np.sum(loaded)) / _FOOT_SENSORS


def _is_standing(info: dict) -> bool:
    feet = info["foot_contact"]
    return (
        float(info["chest_height"]) >= CHEST_UP_HEIGHT
        and bool(feet[0])
        and bool(feet[1])
        and float(info["upright"]) >= FALL_UPRIGHT
    )


def is_stable(info: dict) -> bool:
    if not _is_standing(info):
        return False
    linear = np.asarray(info["base_lin_vel"], dtype=np.float64)
    angular = np.asarray(info["base_ang_vel"], dtype=np.float64)
    return float(np.linalg.norm(linear)) <= STABLE_LIN and float(np.linalg.norm(angular)) <= STABLE_ANG
