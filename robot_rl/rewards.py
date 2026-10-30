"""站立与爬起奖励。

站立课按 Isaac Lab 的速度任务，命令恒为 0。水平速度和转向各用 exp 核对着 0，
再乘以躯干立起来的程度。躺平不动不再得这两项。

getup 和 impact 默认使用本机尺度的恢复/保持目标，允许弱辅助探索。
显式轨迹精修保留 HumanUP 奖励实现。
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

_FOOT_SENSORS = 2


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
    """Robot-scaled recovery objective. Assistance curriculum is in the env;
    reward never consumes a phase, skill label or reference trajectory.
    """
    def __init__(self, weight, control_dt=.02):
        self.control_dt = float(control_dt)
        self.regularization_strength = 1.
        self.adapt_regularization = False
        super().__init__(weight)

    def reset(self):
        super().reset()
        self._previous_potential = None

    def geometry(self, info):
        height = float(np.clip((info["base_height"] - .15) / max(info["stand_base_height"] - .15, .1), 0., 1.))
        upright = float(np.clip(min(info["upright"], info["chest_upright"]), 0., 1.))
        feet = np.asarray(info["floor_n"])[:2]
        load = float(np.clip(np.sum(feet) / self.weight, 0., 1.))
        potential = 4. * height * (.25 + .75 * upright) + upright
        return height, upright, load, potential

    def prime(self, info):
        self._previous_potential = self.geometry(info)[3]

    def end_episode(self, height, stand_height):
        pass

    def update(self, action, info):
        if info.get("diverged") or info.get("reward_failure"):
            # An absorbing failure has zero next potential. A time limit does
            # not: SB3 bootstraps it as a continuation of the physical task.
            value = -10. - (self._previous_potential or 0.)
            return value, {"termination": value, "total": value}
        from robot_rl.recovery import is_recovered
        height, upright, load, potential = self.geometry(info)
        previous = potential if self._previous_potential is None else self._previous_potential
        command = np.asarray(action)
        last = command if self._previous_action is None else self._previous_action
        near_stand = float(np.clip((height - .65) / .3, 0., 1.)) * upright ** 2
        support = float(np.all(np.asarray(info["floor_n"])[:2] > 2.))
        nonfoot = float(np.sum(np.asarray(info["floor_n"])[2:]) / self.weight)
        linear = float(np.sum(np.asarray(info["base_lin_vel"]) ** 2))
        angular = float(np.sum(np.asarray(info["base_ang_vel"]) ** 2))
        still = float(np.exp(-linear / .04 - angular / .25))
        # Below standing there is a persistent deficit, rather than a positive
        # head-height income that can be collected from a seated support trap.
        rates = {
            "height_deficit": -2. * (1. - height),
            "upright_deficit": -.5 * (1. - upright),
            "support_deficit": -near_stand * ((1. - support) + min(nonfoot, 1.)),
            "standing": 6. * near_stand * support * load * np.exp(-10. * nonfoot) * still,
            "stable": 4. * float(is_recovered(info)),
            "lin_vel": -near_stand * min(linear, 4.),
            "ang_vel": -.2 * near_stand * min(angular, 25.),
            "feet_slide": -.5 * near_stand * min(float(info["foot_slip"]), 2.),
            "action_rate": -.005 * float(np.mean((command - last) ** 2)),
            "energy": -.00002 * float(np.linalg.norm(np.asarray(info["torque"]) * info["joint_velocity"])),
            "dof_limit": -.5 * float(np.sum(info["joint_limit_excess"])),
        }
        terms = {key: float(value) * self.control_dt for key, value in rates.items()}
        terms["progress"] = .99 * potential - previous
        self._previous_potential = potential
        self._previous_action = command.copy()
        value = float(sum(terms.values()))
        terms["total"] = value
        info["regularization_strength"] = 1.
        return value, terms


class HumanUPReward(StandReward):
    """HumanUP upstream reward formulas (commit 7516e0f), with joint indices
    mapped to our robot. Heights/distances retain upstream values in metres.
    No additional progress potentials, support gates or success/time bonuses.
    """
    def __init__(self, weight, stage="discovery", trajectory=None, control_dt=.02):
        self.stage, self.trajectory = stage, trajectory
        if not np.isfinite(control_dt) or control_dt <= 0.:
            raise ValueError("control_dt must be positive and finite")
        self.control_dt = float(control_dt)
        self.regularization_strength = 1.
        self.adapt_regularization = True
        self._terminal_height = 0.
        super().__init__(weight)

    def reset(self):
        super().reset()
        self._previous_velocity = None
        self._previous_height = None
        self._previous_force = None

    def prime(self, info):
        self._previous_velocity = np.asarray(info["joint_velocity"]).copy()
        self._previous_height = float(info["base_height"])
        self._previous_force = float(np.linalg.norm(info["feet_force"][:, 2]))

    def end_episode(self, height, stand_height):
        self._terminal_height = float(height)

    def update(self, action, info):
        discovery = self.stage == "discovery"
        if info.get("diverged") or info.get("reward_failure"):
            value = (-500. if discovery else -50.) * self.control_dt
            return value, {"termination": value, "total": value}
        action = np.asarray(action, dtype=float)
        previous = np.zeros_like(action) if self._previous_action is None else self._previous_action
        rate = float(np.linalg.norm(action - previous))
        self._consume_action(action)
        height, head = float(info["base_height"]), float(info["head_height"])
        force = np.asarray(info["feet_force"])
        force_norm = float(np.linalg.norm(force[:, 2]))
        qd, torque = (np.asarray(info[k]) for k in ("joint_velocity", "torque"))
        last_qd = np.zeros_like(qd) if self._previous_velocity is None else self._previous_velocity
        qdd = (qd - last_qd) / self.control_dt
        strength = self.regularization_strength
        distance = float(info["feet_distance"])
        d_min = float(np.clip(distance - .25, -.5, 0.))
        d_max = float(np.clip(distance - 1., 0., 1.))
        # Only names in upstream regularization_names receive the curriculum
        # multiplier. In particular action_rate/torques/ang_vel do not.
        terms = {
            "dof_vel": (-1e-4 if discovery else -1e-3) * strength * float(np.sum(qd ** 2)),
            "action_rate": -.1 * rate,
            "torque": (-1e-6 if discovery else -.003) * float(np.linalg.norm(torque)),
            "ang_vel": -.1 * float(np.sum(np.asarray(info["base_ang_vel"]) ** 2)),
            "lin_vel": -.1 * float(np.linalg.norm(info["base_lin_vel"])),
            "energy": -1e-4 * float(np.linalg.norm(torque * qd)),
            "dof_limit": -5. * strength * float(np.sum(info["joint_limit_excess"])),
            "torque_limit": (-.1 if discovery else -5.) * strength * float(np.sum(info["torque_limit_excess"])),
            "feet_distance": float(np.exp(-100. * abs(d_min)) + np.exp(-100. * abs(d_max))),
            "feet_orientation": -.5 * float(info["feet_tilt"]),
        }
        if discovery:
            right = action[[2, 3, 4, 5, 6, 7, 14, 15, 16]]
            left = action[[8, 9, 10, 11, 12, 13, 17, 18, 19]]
            signs = np.array([-1., -1., 1., 1., 1., -1., 1., -1., 1.])
            standing = head > 1.1
            terms.update({
                "dof_acc": -1e-7 * float(np.sum(qdd ** 2)),
                "feet_slide": -float(np.sum(np.sqrt(info["feet_speed"]) * (force[:, 2] > 5.))),
                "orientation": -strength * float(np.sum(np.asarray(info["projected_gravity"])[:2] ** 2)),
                "height": 5. * float(np.expm1(np.clip(height, 0., .728))),
                "head_height": 5. * float(np.expm1(np.clip(head, 0., 1.3))),
                "d_height": float(height > (height if self._previous_height is None else self._previous_height)),
                "d_feet": float(force_norm > (force_norm if self._previous_force is None else self._previous_force)),
                "feet": 2.5 * strength * float(np.all(np.linalg.norm(force, axis=1) > 2.) and np.all(info["feet_height"] < .1)),
                "upright": .25 * float(np.exp(-info["projected_gravity"][2])),
                "feet_height": 2.5 * strength * float(np.exp(-10. * np.mean(info["feet_height"]))),
                "symmetry": -float(not standing) * float(np.linalg.norm(left - signs * right)),
                "waist_symmetry": -float(not standing) * abs(float(action[0])),
                "standing_pose": -.03 * strength * float(standing) * float(np.sum(np.asarray(info["joint_pos"]) ** 2)),
            })
        else:
            target, _ = self.trajectory.sample(info["tracking_pose"], info["tracking_time"])
            error = float(np.sum((np.asarray(info["joint_q"]) - target) ** 2))
            tracking = float(np.exp(-error / 4.))
            ankles, upper = [6, 7, 12, 13], list(range(14, 20))
            terms.update({
                "ankle_limit": -5. * float(np.sum(info["joint_limit_excess"][ankles])),
                "upper_limit": -5. * float(np.sum(info["joint_limit_excess"][upper])),
                "upper_torque_limit": -5. * float(np.sum(info["torque_limit_excess"][upper])),
                "ankle_torque": -.01 * float(np.linalg.norm(torque[ankles])),
                "upper_torque": -.01 * float(np.linalg.norm(torque[upper])),
                "tracking": 8. * tracking,
            })
            info["motion_match"] = {"tracking": tracking}
        self._previous_velocity = qd.copy()
        self._previous_height, self._previous_force = height, force_norm
        # Upstream updates the curriculum AFTER calculating the current reward.
        if self.adapt_regularization:
            self.regularization_strength = regularization_scale(strength, self._terminal_height)
        terms = {name: float(value) * self.control_dt for name, value in terms.items()}
        value = float(sum(terms.values()))
        terms["total"] = value
        info["regularization_strength"] = strength
        return value, terms


def regularization_scale(strength, terminal_height):
    """Official step_height curriculum, applied after each control step."""
    if terminal_height > .65:
        strength *= 1.0001
    elif terminal_height < .1:
        strength *= .9999
    return float(np.clip(strength, .8, 2.))


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
