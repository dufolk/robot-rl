"""两足人形的关节顺序、站立姿态和底层 PD 增益。

动作是相对站立姿态的归一化关节目标，不是力矩。
`action = 0` 就是这组站立角。`±1` 顶到对应关节限位。
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

FOOT_GEOM_NAMES = ("right_foot_geom", "left_foot_geom")
FOOT_SITE_NAMES = ("right_foot_site", "left_foot_site")
POSES = ("stand", "supine", "prone")


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
    "upright",
    "fallen",
    "foot_contact",
    "foot_pos",
    "com",
    "base_lin_vel",
    "base_ang_vel",
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
        ("foot_contact", 2),
        ("base_height", 1),
    )
    for name, size in fields:
        slices[name] = slice(cursor, cursor + size)
        cursor += size
    return slices


def observation_size(n_joints: int) -> int:
    return observation_slices(n_joints)["base_height"].stop


def describe_observation(n_joints: int | None = None) -> str:
    n = len(JOINTS) if n_joints is None else n_joints
    lines = ["观测向量，float32，机体坐标系："]
    for name, slc in observation_slices(n).items():
        lines.append(f"  [{slc.start:3d}:{slc.stop:<3d}]  {name}")
    lines.append("joint_pos 是相对站立姿态的角度（弧度）。")
    lines.append("foot_contact 顺序是右脚、左脚，0 或 1。")
    lines.append("projected_gravity 直立时约为 (0, 0, -1)。")
    return "\n".join(lines)
