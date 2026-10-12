"""MuJoCo 人形：PD 跟踪、炮弹池、观测和诊断。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robot_rl.robot import (
    CONTROL_HZ,
    DIAGNOSTIC_KEYS,
    FALL_HEIGHT,
    FALL_UPRIGHT,
    FOOT_CLEARANCE,
    FOOT_GEOM_NAMES,
    FOOT_SITE_NAMES,
    JOINTS,
    N_PROJECTILES,
    POSES,
    PROJECTILE_LIFETIME,
    PROJECTILE_MASS,
    PROJECTILE_RADIUS,
    PROJECTILE_SPAWN_OFFSET,
    PROJECTILE_SPEED,
    XML_PATH,
    observation_size,
    observation_slices,
)

_QUAT_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_GRAVITY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class _Ball:
    body_id: int
    geom_id: int
    qpos_adr: int
    dof_adr: int
    park: np.ndarray
    active: bool = False
    expire: float = 0.0


def _require_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"模型里找不到 {name}")
    return index


def load_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    for index in range(N_PROJECTILES):
        park = (8.0 + index * 0.5, 8.0, -6.0)
        body = spec.worldbody.add_body(name=f"ball_{index}", pos=list(park))
        body.add_freejoint(name=f"ball_{index}_joint")
        geom = body.add_geom(
            name=f"ball_{index}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[PROJECTILE_RADIUS, 0.0, 0.0],
            mass=PROJECTILE_MASS,
            rgba=[0.75, 0.42, 0.16, 1.0],
            contype=1,
            conaffinity=1,
            condim=3,
        )
        geom.friction[:] = [0.45, 0.005, 0.0001]
    model = spec.compile()
    return model


class HumanoidSim:
    """一具两足人形。

    `apply_action` 接收 [-1, 1] 的关节目标。0 是站立姿态。
    每个物理步把目标变成力矩：

        tau = kp (q_des - q) - kv qd

    增益取得比较硬，这样在没有策略时零动作也能站住。
    真正的平衡和爬起要改关节目标，不能靠把这层 PD 再加硬。
    """

    def __init__(self) -> None:
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_hz = CONTROL_HZ
        self.decimation = int(round((1.0 / CONTROL_HZ) / self.sim_dt))
        if self.decimation < 1:
            raise RuntimeError("控制频率高于物理步长")

        self.n_act = len(JOINTS)
        self.obs_slices = observation_slices(self.n_act)
        self.obs_size = observation_size(self.n_act)

        joint_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.name) for spec in JOINTS],
            dtype=np.int32,
        )
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        ]
        expected = [spec.name for spec in JOINTS]
        if actuator_names != expected:
            raise ValueError(f"电机顺序应为 {expected}，模型里是 {actuator_names}")

        self.qpos_adr = self.model.jnt_qposadr[joint_ids].copy()
        self.dof_adr = self.model.jnt_dofadr[joint_ids].copy()
        self.q_low = self.model.jnt_range[joint_ids, 0].copy()
        self.q_high = self.model.jnt_range[joint_ids, 1].copy()
        self.default_q = np.array([spec.default for spec in JOINTS], dtype=np.float64)
        self.kp = np.array([spec.kp for spec in JOINTS], dtype=np.float64)
        self.kv = np.array([spec.kv for spec in JOINTS], dtype=np.float64)
        self.torque_limit = np.array([spec.torque_limit for spec in JOINTS], dtype=np.float64)
        if np.any(self.default_q < self.q_low - 1e-6) or np.any(self.default_q > self.q_high + 1e-6):
            raise ValueError("站立姿态超出关节限位")
        span_low = self.default_q - self.q_low
        span_high = self.q_high - self.default_q
        self.action_scale = np.maximum(span_low, span_high)

        self.root_qpos_adr = int(self.model.jnt_qposadr[_require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")])
        self.pelvis_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor_geom = _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOM_NAMES],
            dtype=np.int32,
        )
        self.foot_site_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES],
            dtype=np.int32,
        )
        self._robot_geoms = self._collect_robot_geoms()
        self._balls = self._collect_balls()

        self.q_target = self.default_q.copy()
        self.last_action = np.zeros(self.n_act, dtype=np.float64)
        self.foot_contact = np.zeros(2, dtype=bool)
        self.last_hit_force = 0.0
        self.pose = "stand"
        self.diverged = False
        self._velocity = np.zeros((6, 1), dtype=np.float64)
        self._contact_force = np.zeros((6, 1), dtype=np.float64)
        self.reset("stand")

    def reset(self, pose: str = "stand") -> None:
        if pose not in POSES:
            raise ValueError(f"未知姿态 {pose}，可选 {POSES}")
        mujoco.mj_resetData(self.model, self.data)
        self.pose = pose
        self.diverged = False
        self.last_hit_force = 0.0
        self.foot_contact[:] = False

        root = self.root_qpos_adr
        self.data.qpos[root : root + 3] = (0.0, 0.0, 0.45 if pose != "stand" else 1.0)
        self.data.qpos[root + 3 : root + 7] = _root_quat(pose)
        self.data.qpos[self.qpos_adr] = self.default_q
        self.data.qvel[:] = 0.0

        for ball in self._balls:
            self._deactivate(ball)
        mujoco.mj_forward(self.model, self.data)
        self._plant_on_ground()
        self.apply_action(np.zeros(self.n_act, dtype=np.float64))

    def apply_action(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (self.n_act,):
            raise ValueError(f"动作维度应为 {self.n_act}，收到 {command.shape}")
        command = np.clip(command, -1.0, 1.0)
        self.last_action[:] = command
        target = self.default_q + command * self.action_scale
        self.q_target[:] = np.clip(target, self.q_low, self.q_high)

    def spawn_projectile(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        speed: float = PROJECTILE_SPEED,
    ) -> bool:
        aim = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(aim))
        if norm < 1e-8 or not math.isfinite(speed) or speed <= 0.0:
            return False
        aim /= norm
        start = np.asarray(origin, dtype=np.float64).reshape(3) + aim * PROJECTILE_SPAWN_OFFSET
        ball = self._allocate_ball()
        self._activate(ball, start, aim * float(speed))
        mujoco.mj_forward(self.model, self.data)
        return True

    def physics_step(self) -> None:
        if self.diverged:
            return
        self._apply_pd()
        self._pin_inactive()
        mujoco.mj_step(self.model, self.data)
        self._retire_projectiles()
        self._read_contacts()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.diverged = True

    def observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_size, dtype=np.float32)
        slices = self.obs_slices
        linear, angular = self._base_velocity()
        gravity = self._projected_gravity()
        obs[slices["base_lin_vel"]] = linear
        obs[slices["base_ang_vel"]] = angular
        obs[slices["projected_gravity"]] = gravity
        joint_pos = self.data.qpos[self.qpos_adr] - self.default_q
        obs[slices["joint_pos"]] = joint_pos
        obs[slices["joint_vel"]] = self.data.qvel[self.dof_adr]
        obs[slices["last_action"]] = self.last_action
        obs[slices["foot_contact"]] = self.foot_contact.astype(np.float32)
        obs[slices["base_height"]] = self.data.xpos[self.pelvis_id, 2]
        return obs

    def diagnostics(self) -> dict:
        linear, angular = self._base_velocity()
        gravity = self._projected_gravity()
        height = float(self.data.xpos[self.pelvis_id, 2])
        upright = float(-gravity[2])
        fallen = bool(height < FALL_HEIGHT or upright < FALL_UPRIGHT)
        info = {
            "base_height": height,
            "upright": upright,
            "fallen": fallen,
            "foot_contact": (bool(self.foot_contact[0]), bool(self.foot_contact[1])),
            "foot_pos": self.data.site_xpos[self.foot_site_ids].copy(),
            "com": self.data.subtree_com[self.pelvis_id].copy(),
            "base_lin_vel": linear.copy(),
            "base_ang_vel": angular.copy(),
            "projected_gravity": gravity.copy(),
            "hit_force": float(self.last_hit_force),
            "pose": self.pose,
            "diverged": self.diverged,
        }
        missing = [key for key in DIAGNOSTIC_KEYS if key not in info]
        if missing:
            raise RuntimeError(f"诊断缺字段 {missing}")
        return info

    def lowest_point_z(self) -> float:
        lowest = math.inf
        for geom_id in self._robot_geoms:
            lowest = min(lowest, self._geom_min_z(int(geom_id)))
        return float(lowest)

    def _apply_pd(self) -> None:
        position = self.data.qpos[self.qpos_adr]
        velocity = self.data.qvel[self.dof_adr]
        torque = self.kp * (self.q_target - position) - self.kv * velocity
        np.clip(torque, -self.torque_limit, self.torque_limit, out=torque)
        self.data.ctrl[:] = torque

    def _plant_on_ground(self) -> None:
        lowest = self.lowest_point_z()
        self.data.qpos[self.root_qpos_adr + 2] -= lowest - FOOT_CLEARANCE
        mujoco.mj_forward(self.model, self.data)

    def _base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_id,
            self._velocity,
            1,
        )
        angular = self._velocity[:3, 0].copy()
        linear = self._velocity[3:, 0].copy()
        return linear, angular

    def _projected_gravity(self) -> np.ndarray:
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        return rotation.T @ _GRAVITY_DIR

    def _read_contacts(self) -> None:
        feet = np.zeros(2, dtype=bool)
        hit = 0.0
        ball_geoms = {ball.geom_id for ball in self._balls if ball.active}
        robot_geoms = set(self._robot_geoms)
        foot_ids = [int(gid) for gid in self.foot_geom_ids]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            pair_set = set(pair)
            if self.floor_geom in pair_set:
                for side, geom_id in enumerate(foot_ids):
                    if geom_id in pair_set:
                        feet[side] = True
            if pair_set & ball_geoms and pair_set & robot_geoms:
                mujoco.mj_contactForce(self.model, self.data, index, self._contact_force)
                hit = max(hit, abs(float(self._contact_force[0, 0])))
        self.foot_contact = feet
        self.last_hit_force = hit

    def _collect_robot_geoms(self) -> np.ndarray:
        parent = self.model.body_parentid
        pelvis = self.pelvis_id

        def is_robot(body_id: int) -> bool:
            body = int(body_id)
            while body != 0:
                if body == pelvis:
                    return True
                body = int(parent[body])
            return False

        geoms = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_contype[geom_id]) == 0:
                continue
            if is_robot(int(self.model.geom_bodyid[geom_id])):
                geoms.append(geom_id)
        return np.array(geoms, dtype=np.int32)

    def _collect_balls(self) -> list[_Ball]:
        balls = []
        for index in range(N_PROJECTILES):
            joint_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ball_{index}_joint")
            geom_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ball_{index}_geom")
            body_id = int(self.model.jnt_bodyid[joint_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            park = self.model.qpos0[qpos_adr : qpos_adr + 3].copy()
            balls.append(
                _Ball(
                    body_id=body_id,
                    geom_id=geom_id,
                    qpos_adr=qpos_adr,
                    dof_adr=int(self.model.jnt_dofadr[joint_id]),
                    park=park,
                )
            )
        return balls

    def _allocate_ball(self) -> _Ball:
        for ball in self._balls:
            if not ball.active:
                return ball
        return min(self._balls, key=lambda item: item.expire)

    def _activate(self, ball: _Ball, origin: np.ndarray, velocity: np.ndarray) -> None:
        ball.active = True
        ball.expire = float(self.data.time) + PROJECTILE_LIFETIME
        self.model.geom_contype[ball.geom_id] = 1
        self.model.geom_conaffinity[ball.geom_id] = 1
        self.model.body_gravcomp[ball.body_id] = 0.0
        adr = ball.qpos_adr
        self.data.qpos[adr : adr + 3] = origin
        self.data.qpos[adr + 3 : adr + 7] = _QUAT_IDENTITY
        self.data.qvel[ball.dof_adr : ball.dof_adr + 3] = velocity
        self.data.qvel[ball.dof_adr + 3 : ball.dof_adr + 6] = 0.0

    def _deactivate(self, ball: _Ball) -> None:
        ball.active = False
        ball.expire = 0.0
        self.model.geom_contype[ball.geom_id] = 0
        self.model.geom_conaffinity[ball.geom_id] = 0
        self.model.body_gravcomp[ball.body_id] = 1.0
        self._write_park(ball)

    def _write_park(self, ball: _Ball) -> None:
        adr = ball.qpos_adr
        self.data.qpos[adr : adr + 3] = ball.park
        self.data.qpos[adr + 3 : adr + 7] = _QUAT_IDENTITY
        self.data.qvel[ball.dof_adr : ball.dof_adr + 6] = 0.0

    def _pin_inactive(self) -> None:
        for ball in self._balls:
            if not ball.active:
                self._write_park(ball)

    def _retire_projectiles(self) -> None:
        now = float(self.data.time)
        for ball in self._balls:
            if not ball.active:
                continue
            height = float(self.data.qpos[ball.qpos_adr + 2])
            if now >= ball.expire or height < -1.0:
                self._deactivate(ball)

    def _geom_min_z(self, geom_id: int) -> float:
        kind = int(self.model.geom_type[geom_id])
        center = self.data.geom_xpos[geom_id]
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        size = self.model.geom_size[geom_id]
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(center[2] - size[0])
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            lowest = math.inf
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = center + rotation @ (np.array([sx, sy, sz]) * size)
                        lowest = min(lowest, float(corner[2]))
            return float(lowest)
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = float(size[1])
            radius = float(size[0])
            axis = rotation[:, 2]
            end_a = center + axis * half
            end_b = center - axis * half
            return float(min(end_a[2], end_b[2]) - radius)
        return float(center[2])


def _root_quat(pose: str) -> np.ndarray:
    if pose == "stand":
        return _QUAT_IDENTITY.copy()
    # 绕 Y 轴 ±90°：仰卧胸口朝上，俯卧胸口朝下。
    half = -math.pi / 4.0 if pose == "supine" else math.pi / 4.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)
