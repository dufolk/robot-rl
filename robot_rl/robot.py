"""两足人形的关节顺序、站立姿态和底层 PD 增益。

动作是相对站立姿态的归一化关节目标，不是力矩。
`action = 0` 就是这组站立角。新版 HumanUP 用正负两侧各自的活动范围，`±1` 对应两侧限位；历史模型保留旧映射。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

XML_PATH = Path(__file__).resolve().parent / "assets" / "humanoid.xml"

CONTROL_HZ = 50
FALL_HEIGHT = 0.45
FALL_UPRIGHT = 0.4
FOOT_CLEARANCE = 0.006

N_PROJECTILES = 12
PROJECTILE_MASS = 3.0
PROJECTILE_RADIUS = 0.07
PROJECTILE_SPEED = 22.0
PROJECTILE_LIFETIME = 4.0
PROJECTILE_SPAWN_OFFSET = 0.4
# 查看器左键，也是 impact 课的 1 倍动量。5 倍只加质量，速度保持这一档，避免一步穿过身体。
SHOT_MASS = 48.0
SHOT_SPEED = 40.0

FOOT_GEOM_NAMES = ("right_foot_geom", "left_foot_geom")
FOOT_SITE_NAMES = ("right_foot_site", "left_foot_site")
# 会碰上地面或炮弹的部位。顺序就是观测里 pressure 的顺序。
PRESSURE_SENSORS = (
    "right_foot_geom",
    "left_foot_geom",
    "right_shin",
    "left_shin",
    "right_thigh",
    "left_thigh",
    "right_hand",
    "left_hand",
    "right_forearm",
    "left_forearm",
    "right_upper_arm",
    "left_upper_arm",
    "pelvis",
    "chest",
    "head",
)
POSES = ("stand", "supine", "prone", "random")


@dataclass(frozen=True)
class JointSpec:
    name: str
    default: float
    kp: float
    kv: float
    torque_limit: float


# 顺序同时是动作、观测和电机 ctrl 的顺序。
JOINTS: tuple[JointSpec, ...] = (
    JointSpec("waist_yaw", 0.0, 480.0, 18.0, 80.0),
    JointSpec("waist_pitch", 0.08, 1120.0, 36.0, 180.0),
    JointSpec("right_hip_yaw", 0.0, 320.0, 12.0, 80.0),
    JointSpec("right_hip_roll", -0.06, 960.0, 30.0, 180.0),
    JointSpec("right_hip_pitch", -0.25, 1280.0, 42.0, 250.0),
    JointSpec("right_knee", 0.50, 1440.0, 42.0, 280.0),
    JointSpec("right_ankle_pitch", -0.25, 640.0, 18.0, 160.0),
    JointSpec("right_ankle_roll", 0.06, 400.0, 12.0, 100.0),
    JointSpec("left_hip_yaw", 0.0, 320.0, 12.0, 80.0),
    JointSpec("left_hip_roll", 0.06, 960.0, 30.0, 180.0),
    JointSpec("left_hip_pitch", -0.25, 1280.0, 42.0, 250.0),
    JointSpec("left_knee", 0.50, 1440.0, 42.0, 280.0),
    JointSpec("left_ankle_pitch", -0.25, 640.0, 18.0, 160.0),
    JointSpec("left_ankle_roll", -0.06, 400.0, 12.0, 100.0),
    JointSpec("right_shoulder_pitch", -0.35, 240.0, 9.0, 40.0),
    JointSpec("right_shoulder_roll", -0.25, 192.0, 6.0, 35.0),
    JointSpec("right_elbow", 0.55, 192.0, 6.0, 35.0),
    JointSpec("left_shoulder_pitch", -0.35, 240.0, 9.0, 40.0),
    JointSpec("left_shoulder_roll", 0.25, 192.0, 6.0, 35.0),
    JointSpec("left_elbow", 0.55, 192.0, 6.0, 35.0),
)

DIAGNOSTIC_KEYS = (
    "base_height",
    "chest_height",
    "up_velocity",
    "upright",
    "fallen",
    "foot_contact",
    "pressure",
    "foot_pos",
    "com",
    "base_lin_vel",
    "base_ang_vel",
    "yaw_rate",
    "body_pos_err",
    "body_rot_err",
    "body_align",
    "body_height",
    "foot_slip",
    "joint_pos",
    "joint_acc",
    "torque",
    "projected_gravity",
    "hit_force",
    "pose",
    "diverged",
)


def observation_slices(n_joints: int) -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    fields = (
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", n_joints),
        ("joint_vel", n_joints),
        ("last_action", n_joints),
        ("pressure", len(PRESSURE_SENSORS)),
        ("base_height", 1),
        ("home", 4),
    )
    for name, size in fields:
        slices[name] = slice(cursor, cursor + size)
        cursor += size
    return slices


def observation_size(n_joints: int) -> int:
    return observation_slices(n_joints)["home"].stop


def describe_observation(n_joints: int | None = None) -> str:
    n = len(JOINTS) if n_joints is None else n_joints
    lines = ["观测向量，float32，机体坐标系："]
    for name, slc in observation_slices(n).items():
        lines.append(f"  [{slc.start:3d}:{slc.stop:<3d}]  {name}")
    lines.append("joint_pos 是相对站立姿态的角度（弧度）。")
    lines.append("pressure 是各部位法向接触力除以体重。顺序：")
    lines.append("  " + ", ".join(PRESSURE_SENSORS))
    lines.append("站稳时两只脚大约各 0.5。projected_gravity 直立时约为 (0, 0, -1)。")
    lines.append("home 是零动作原点站姿：机体坐标下原点在前方、左方多远（米），以及朝向差的 sin、cos。")
    lines.append("站在原点且朝向一致时 home 是 (0, 0, 0, 1)。")
    return "\n".join(lines)
