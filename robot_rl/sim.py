"""MuJoCo 人形：PD 跟踪、炮弹池、观测和诊断。"""

from __future__ import annotations

import math
import os
import tempfile
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
    PRESSURE_SENSORS,
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
# 任意姿态里一半贴地，一半悬在 0.8–3 米并给下落速度。
_RANDOM_AIR_FRACTION = 0.5
_RANDOM_DROP_HEIGHT = (0.8, 3.0)


@dataclass
class _Ball:
    body_id: int
    geom_id: int
    qpos_adr: int
    dof_adr: int
    park: np.ndarray
    nominal_mass: float
    nominal_inertia: np.ndarray
    active: bool = False
    expire: float = 0.0


def _heading(xmat: np.ndarray) -> float:
    rotation = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _require_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"模型里找不到 {name}")
    return index


_MODEL_SNAPSHOT: str | None = None


def load_model() -> mujoco.MjModel:
    """每个调用返回一份独立模型。炮弹会改质量和对碰开关，不能共用同一份。"""
    global _MODEL_SNAPSHOT
    if _MODEL_SNAPSHOT is None:
        fd, path = tempfile.mkstemp(prefix="humanoid-", suffix=".mjb")
        os.close(fd)
        mujoco.mj_saveModel(_compile_model(), path, None)
        _MODEL_SNAPSHOT = path
    return mujoco.MjModel.from_binary_path(_MODEL_SNAPSHOT)


def _compile_model() -> mujoco.MjModel:
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

    def __init__(self, physics_dt: float | None = None, action_mapping: str = "legacy") -> None:
        if action_mapping not in ("legacy", "joint_limits"):
            raise ValueError("action_mapping must be legacy or joint_limits")
        self.action_mapping = action_mapping
        self.model = load_model()
        if physics_dt is not None:
            if not math.isfinite(physics_dt) or physics_dt <= 0.:
                raise ValueError("physics_dt must be positive and finite")
            self.model.opt.timestep = float(physics_dt)
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
        self.action_scale_low, self.action_scale_high = span_low, span_high
        self.action_scale = np.maximum(span_low, span_high)

        root_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos_adr = int(self.model.jnt_qposadr[root_id])
        self.root_dof_adr = int(self.model.jnt_dofadr[root_id])
        self.pelvis_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.chest_geom_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "chest")
        self.pelvis_geom_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis")
        self._torso_geom_ids = {self.pelvis_geom_id, self.chest_geom_id}
        self.thigh_geom_ids = np.array(
            [
                _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_thigh"),
                _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_thigh"),
            ],
            dtype=np.int32,
        )
        self.floor_geom = _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOM_NAMES],
            dtype=np.int32,
        )
        self.foot_site_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITE_NAMES],
            dtype=np.int32,
        )
        self.pressure_geom_ids = np.array(
            [_require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in PRESSURE_SENSORS],
            dtype=np.int32,
        )
        self._pressure_index = {int(geom_id): index for index, geom_id in enumerate(self.pressure_geom_ids)}
        self._robot_geoms = self._collect_robot_geoms()
        self._robot_geom_set = frozenset(map(int, self._robot_geoms))
        self._foot_side = {int(gid): side for side, gid in enumerate(self.foot_geom_ids)}
        missing = [name for name in self._colliding_geom_names() if name not in PRESSURE_SENSORS]
        if missing:
            raise RuntimeError(f"这些碰撞体没有压力传感器：{missing}")
        weight = float(self.model.body_subtreemass[self.pelvis_id]) * 9.81
        if weight <= 0.0:
            raise RuntimeError("体重应该是正的")
        self.weight = weight
        self._balls = self._collect_balls()
        self._park_cache_state = None
        self.robot_body_ids = self._collect_robot_bodies()
        self.robot_body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) for body_id in self.robot_body_ids
        ]
        if any(name is None for name in self.robot_body_names):
            raise RuntimeError("骨盆下面有没命名的身体部件")
        self.ref_body_height: dict[str, float] = {}
        self.foot_body_ids = self.model.geom_bodyid[self.foot_geom_ids].copy()
        self.ref_body_pos: np.ndarray | None = None
        self.ref_body_quat: np.ndarray | None = None
        self.ref_pelvis_pos = np.zeros(3, dtype=np.float64)
        self.ref_yaw = 0.0

        self.q_target = self.default_q.copy()
        self.last_action = np.zeros(self.n_act, dtype=np.float64)
        self.foot_contact = np.zeros(2, dtype=bool)
        self.foot_slip = 0.0
        self.pressure = np.zeros(len(PRESSURE_SENSORS), dtype=np.float64)
        self.torso_floor_n = 0.0
        self.pelvis_floor_n = 0.0
        self.thigh_floor_n = np.zeros(2, dtype=np.float64)
        self.floor_n = np.zeros(len(PRESSURE_SENSORS), dtype=np.float64)
        self._pelvis_slot = PRESSURE_SENSORS.index("pelvis")
        self._chest_slot = PRESSURE_SENSORS.index("chest")
        self._thigh_slots = (
            PRESSURE_SENSORS.index("right_thigh"),
            PRESSURE_SENSORS.index("left_thigh"),
        )
        self.last_hit_force = 0.0
        self._assist_n = 0.0
        self.pose = "stand"
        self.diverged = False
        self._velocity = np.zeros((6, 1), dtype=np.float64)
        self._contact_force = np.zeros((6, 1), dtype=np.float64)
        self.reset("stand")

    def reset(self, pose: str = "stand", rng: np.random.Generator | None = None) -> None:
        if pose not in POSES:
            raise ValueError(f"未知姿态 {pose}，可选 {POSES}")
        if pose == "random" and rng is None:
            raise ValueError("随机姿态需要随机数发生器")
        mujoco.mj_resetData(self.model, self.data)
        self.pose = pose
        self.diverged = False
        self.last_hit_force = 0.0
        self.foot_contact[:] = False
        self.foot_slip = 0.0
        self.pressure[:] = 0.0

        root = self.root_qpos_adr
        airborne = False
        if pose == "random":
            self._write_random_pose(rng)
            airborne = float(rng.random()) < _RANDOM_AIR_FRACTION
        else:
            self.data.qpos[root : root + 3] = (0.0, 0.0, 0.45 if pose != "stand" else 1.0)
            self.data.qpos[root + 3 : root + 7] = _root_quat(pose)
            self.data.qpos[self.qpos_adr] = self.default_q
        self.data.qvel[:] = 0.0

        for ball in self._balls:
            self._deactivate(ball)
        mujoco.mj_forward(self.model, self.data)
        self._plant_on_ground()
        if airborne:
            self._drop_from_height(rng)
        self.apply_action(np.zeros(self.n_act, dtype=np.float64))
        if pose == "stand" and self.ref_body_pos is None:
            self._capture_stand_reference()

    def apply_action(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.shape != (self.n_act,):
            raise ValueError(f"动作维度应为 {self.n_act}，收到 {command.shape}")
        command = np.clip(command, -1.0, 1.0)
        self.pressure[:] = 0.0
        self.torso_floor_n = 0.0
        self.pelvis_floor_n = 0.0
        self.thigh_floor_n[:] = 0.0
        self.floor_n[:] = 0.0
        self.foot_slip = 0.0
        self.last_action[:] = command
        self.q_target[:] = self.action_targets(command)

    def action_targets(self, action):
        """Map normalized targets; new policies use each side's actual range."""
        command = np.clip(np.asarray(action), -1., 1.)
        scale = (np.where(command >= 0., self.action_scale_high, self.action_scale_low)
                 if self.action_mapping == "joint_limits" else self.action_scale)
        return np.clip(self.default_q + command * scale, self.q_low, self.q_high)

    def spawn_projectile(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        speed: float = PROJECTILE_SPEED,
        mass: float | None = None,
        gap: float | None = None,
    ) -> bool:
        aim = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(aim))
        if norm < 1e-8 or not math.isfinite(speed) or speed <= 0.0:
            return False
        if mass is not None and (not math.isfinite(mass) or mass <= 0.0):
            return False
        aim /= norm
        clearance = PROJECTILE_SPAWN_OFFSET if gap is None else float(gap)
        start = np.asarray(origin, dtype=np.float64).reshape(3) + aim * clearance
        ball = self._allocate_ball()
        self._set_ball_mass(ball, ball.nominal_mass if mass is None else float(mass))
        self._activate(ball, start, aim * float(speed))
        mujoco.mj_forward(self.model, self.data)
        return True

    def set_assist(self, newtons: float, upright_only: bool = True) -> None:
        """爬起时加在骨盆上的向上拉力。躯干没立起来就不施力。"""
        self._assist_n = max(0.0, float(newtons))
        self._assist_upright_only = upright_only

    def physics_step(self) -> None:
        if self.diverged:
            return
        self._apply_pd()
        self._apply_assist()
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
        obs[slices["pressure"]] = self.pressure
        obs[slices["base_height"]] = self.data.xpos[self.pelvis_id, 2]
        obs[slices["home"]] = self._home_observation()
        return obs

    def control_state(self) -> dict:
        """训练一步要用的量。不含每个身体部件的对齐字典。"""
        linear, angular = self._base_velocity()
        gravity = self._projected_gravity()
        height = float(self.data.xpos[self.pelvis_id, 2])
        upright = float(-gravity[2])
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        world_lin = rotation @ linear
        world_ang = rotation @ angular
        return {
            "base_height": height,
            "chest_height": float(self.data.geom_xpos[self.chest_geom_id, 2]),
            "up_velocity": float(world_lin[2]),
            "upright": upright,
            "fallen": bool(height < FALL_HEIGHT or upright < FALL_UPRIGHT),
            "foot_contact": (bool(self.foot_contact[0]), bool(self.foot_contact[1])),
            "base_lin_vel": linear.copy(),
            "base_ang_vel": angular.copy(),
            "yaw_rate": float(world_ang[2]),
            "foot_slip": float(self.foot_slip),
            "joint_pos": (self.data.qpos[self.qpos_adr] - self.default_q).copy(),
            "joint_acc": self.data.qacc[self.dof_adr].copy(),
            "torque": self.data.ctrl.copy(),
            "projected_gravity": gravity.copy(),
            "torso_floor_n": float(self.torso_floor_n),
            "pelvis_floor_n": float(self.pelvis_floor_n),
            "thigh_floor_n": self.thigh_floor_n.copy(),
            "floor_n": self.floor_n.copy(),
            "hit_force": float(self.last_hit_force),
            "pose": self.pose,
            "diverged": self.diverged,
        }

    def diagnostics(self) -> dict:
        info = self.control_state()
        info["pressure"] = self.pressure.copy()
        info["foot_pos"] = self.data.site_xpos[self.foot_site_ids].copy()
        info["com"] = self.data.subtree_com[self.pelvis_id].copy()
        info["body_pos_err"] = self._body_pos_err()
        info["body_rot_err"] = self._body_rot_err()
        info["body_align"] = self._body_align()
        info["body_height"] = dict(self.ref_body_height)
        missing = [key for key in DIAGNOSTIC_KEYS if key not in info]
        if missing:
            raise RuntimeError(f"诊断缺字段 {missing}")
        return info

    def lowest_point_z(self) -> float:
        lowest = math.inf
        for geom_id in self._robot_geoms:
            lowest = min(lowest, self._geom_min_z(int(geom_id)))
        return float(lowest)

    def _write_random_pose(self, rng: np.random.Generator) -> None:
        yaw = float(rng.uniform(-math.pi, math.pi))
        tilt = float(rng.uniform(0.0, math.pi))
        axis_angle = float(rng.uniform(-math.pi, math.pi))
        axis = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0], dtype=np.float64)
        quat = _quat_mul(_quat_axis(np.array([0.0, 0.0, 1.0]), yaw), _quat_axis(axis, tilt))
        root = self.root_qpos_adr
        self.data.qpos[root : root + 3] = (0.0, 0.0, 1.2)
        self.data.qpos[root + 3 : root + 7] = quat
        self.data.qpos[self.qpos_adr] = rng.uniform(self.q_low, self.q_high)

    def _capture_stand_reference(self) -> None:
        self.ref_body_pos = self.data.xpos[self.robot_body_ids].copy()
        self.ref_body_quat = self.data.xquat[self.robot_body_ids].copy()
        self.ref_body_height = {
            name: float(self.ref_body_pos[index, 2]) for index, name in enumerate(self.robot_body_names)
        }
        self.ref_pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        self.ref_yaw = _heading(self.data.xmat[self.pelvis_id])

    def _body_pos_err(self) -> float:
        if self.ref_body_pos is None:
            raise RuntimeError("还没有零动作站立参考")
        delta = self.data.xpos[self.robot_body_ids] - self.ref_body_pos
        return float(np.mean(np.sum(delta * delta, axis=1)))

    def _body_rot_err(self) -> float:
        if self.ref_body_quat is None:
            raise RuntimeError("还没有零动作站立参考")
        dots = np.sum(self.data.xquat[self.robot_body_ids] * self.ref_body_quat, axis=1)
        return float(np.mean(1.0 - np.square(np.clip(dots, -1.0, 1.0))))

    def _body_align(self) -> dict[str, float]:
        if self.ref_body_pos is None or self.ref_body_quat is None:
            raise RuntimeError("还没有零动作站立参考")
        delta = self.data.xpos[self.robot_body_ids] - self.ref_body_pos
        pos_err = np.sum(delta * delta, axis=1)
        dots = np.sum(self.data.xquat[self.robot_body_ids] * self.ref_body_quat, axis=1)
        rot_err = 1.0 - np.square(np.clip(dots, -1.0, 1.0))
        return {
            name: float(pos_err[index] + rot_err[index]) for index, name in enumerate(self.robot_body_names)
        }

    def _home_observation(self) -> np.ndarray:
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        delta = rotation.T @ (self.ref_pelvis_pos - self.data.xpos[self.pelvis_id])
        err = _wrap(self.ref_yaw - _heading(self.data.xmat[self.pelvis_id]))
        return np.array([delta[0], delta[1], math.sin(err), math.cos(err)], dtype=np.float64)

    def _horizontal_speed(self, body_id: int) -> float:
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            self._velocity,
            0,
        )
        linear = self._velocity[3:, 0]
        return float(np.hypot(linear[0], linear[1]))

    def _collect_robot_bodies(self) -> np.ndarray:
        parent = self.model.body_parentid
        pelvis = self.pelvis_id
        bodies = []
        for body_id in range(self.model.nbody):
            body = int(body_id)
            while body != 0:
                if body == pelvis:
                    bodies.append(int(body_id))
                    break
                body = int(parent[body])
        if not bodies:
            raise RuntimeError("没有找到骨盆下面的身体部件")
        return np.array(bodies, dtype=np.int32)

    def _apply_assist(self) -> None:
        self.data.xfrc_applied[:] = 0.0
        if self._assist_n <= 0.0:
            return
        upright = -float(self._projected_gravity()[2])
        if upright < 0.7 and getattr(self, "_assist_upright_only", True):
            return
        self.data.xfrc_applied[self.pelvis_id, 2] = self._assist_n

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

    def _drop_from_height(self, rng: np.random.Generator) -> None:
        extra = float(rng.uniform(*_RANDOM_DROP_HEIGHT))
        self.data.qpos[self.root_qpos_adr + 2] += extra
        adr = self.root_dof_adr
        self.data.qvel[adr : adr + 3] = (
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-3.0, -0.5)),
        )
        self.data.qvel[adr + 3 : adr + 6] = rng.uniform(-2.0, 2.0, size=3)
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
        sensed = np.zeros(len(PRESSURE_SENSORS), dtype=np.float64)
        hit = 0.0
        balls = {ball.geom_id: ball for ball in self._balls if ball.active}
        robot_geoms = self._robot_geom_set
        struck = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            on_floor = self.floor_geom in pair
            if on_floor:
                geom_id = pair[1] if pair[0] == self.floor_geom else pair[0]
                side = self._foot_side.get(geom_id)
                if side is not None and not feet[side]:
                    feet[side] = True
                    self.foot_slip = max(self.foot_slip, self._horizontal_speed(int(self.foot_body_ids[side])))
            sensed_here = [slot for geom_id in pair if (slot := self._pressure_index.get(geom_id)) is not None]
            ball = balls.get(pair[0]) or balls.get(pair[1])
            on_body = pair[0] in robot_geoms or pair[1] in robot_geoms
            if not sensed_here and not (ball is not None and on_body):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self._contact_force)
            newtons = abs(float(self._contact_force[0, 0]))
            normal = newtons / self.weight
            for slot in sensed_here:
                sensed[slot] += normal
            if on_floor:
                for geom_id in pair:
                    slot = self._pressure_index.get(geom_id)
                    if slot is not None:
                        self.floor_n[slot] = max(self.floor_n[slot], newtons)
            if ball is not None and on_body:
                hit = max(hit, normal * self.weight)
                struck.append(ball)
        self.foot_contact = feet
        self.pelvis_floor_n = float(self.floor_n[self._pelvis_slot])
        self.torso_floor_n = float(max(self.floor_n[self._pelvis_slot], self.floor_n[self._chest_slot]))
        self.thigh_floor_n[0] = self.floor_n[self._thigh_slots[0]]
        self.thigh_floor_n[1] = self.floor_n[self._thigh_slots[1]]
        self.pressure = np.maximum(self.pressure, sensed)
        self.last_hit_force = hit
        for ball in struck:
            self._deactivate(ball)

    def _colliding_geom_names(self) -> list[str]:
        names = []
        for geom_id in self._robot_geoms:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            if name:
                names.append(name)
        return names

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
                    nominal_mass=float(self.model.body_mass[body_id]),
                    nominal_inertia=self.model.body_inertia[body_id].copy(),
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
        self._set_ball_mass(ball, ball.nominal_mass)
        self._write_park(ball)

    def _set_ball_mass(self, ball: _Ball, mass: float) -> None:
        # mj_setConst 会把 qpos 复位到 qpos0。改质量前后要把姿态留住。
        if abs(float(self.model.body_mass[ball.body_id]) - mass) < 1e-9:
            return
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ctrl = self.data.ctrl.copy()
        warmstart = self.data.qacc_warmstart.copy()
        act = self.data.act.copy() if self.model.na else None
        time = float(self.data.time)
        self.model.body_mass[ball.body_id] = mass
        self.model.body_inertia[ball.body_id] = ball.nominal_inertia * (mass / ball.nominal_mass)
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self.data.qacc_warmstart[:] = warmstart
        if act is not None:
            self.data.act[:] = act
        self.data.time = time

    def _write_park(self, ball: _Ball) -> None:
        adr = ball.qpos_adr
        self.data.qpos[adr : adr + 3] = ball.park
        self.data.qpos[adr + 3 : adr + 7] = _QUAT_IDENTITY
        self.data.qvel[ball.dof_adr : ball.dof_adr + 6] = 0.0

    def _pin_inactive(self) -> None:
        state = tuple(ball.active for ball in self._balls)
        if state != self._park_cache_state:
            inactive = [ball for ball in self._balls if not ball.active]
            self._park_qpos = np.array([ball.qpos_adr + i for ball in inactive for i in range(7)], dtype=int)
            self._park_values = np.array([value for ball in inactive for value in (*ball.park, *_QUAT_IDENTITY)])
            self._park_qvel = np.array([ball.dof_adr + i for ball in inactive for i in range(6)], dtype=int)
            self._park_cache_state = state
        self.data.qpos[self._park_qpos] = self._park_values
        self.data.qvel[self._park_qvel] = 0.

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


def _quat_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    half = angle * 0.5
    scale = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale],
        dtype=np.float64,
    )


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _root_quat(pose: str) -> np.ndarray:
    if pose == "stand":
        return _QUAT_IDENTITY.copy()
    # 绕 Y 轴 ±90°：仰卧胸口朝上，俯卧胸口朝下。
    half = -math.pi / 4.0 if pose == "supine" else math.pi / 4.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)
