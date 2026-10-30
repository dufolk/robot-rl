"""Gymnasium 课程环境。

stand 使用站立奖励；getup 和 impact 默认使用本项目恢复奖励，可选轨迹精修。
默认发现阶段没有参考时钟；可选精修阶段的时钟仅供奖励使用。
两课的输入保留 189 维物理/动作/接触状态并追加三步真实历史，共 384 维。
稳定两秒后继续保持三秒，固定局长结束时评估成败；躯干接地不提前结束起身。
impact 每局站立开场发出一球，扛住也可成功。

动作 [-1, 1] 是 20 个关节的 PD 目标，0 对应模型的默认关节角。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_rl.curriculum import Curriculum, get_curriculum
from robot_rl.rewards import (
    HumanUPReward,
    GetupReward,
    TORSO_FORCE,
    StandReward,
    _is_standing,
)
from robot_rl.sim import HumanoidSim
from robot_rl.humanup import ObservationHistory, DiscoveryTrajectory, DiscoveryPoseCurriculum, GetupCurriculum, observation_size, PHYSICS_DT, ACTION_MAPPING
from robot_rl.recovery import RecoveryGoal
from robot_rl.strikes import Strike, StrikeSchedule, impact_ray


class HumanoidBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
        curriculum: str | None = None,
        humanup_stage: str = "discovery",
        reference_path: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("交互观看请运行 python -m robot_rl。这里的 render_mode 只用 rgb_array。")
        explicit_horizon = max_episode_steps is not None
        if humanup_stage not in ("discovery", "refinement"):
            raise ValueError("humanup_stage must be discovery or refinement")
        self.humanup_stage = humanup_stage
        self.render_mode = render_mode
        self.curriculum: Curriculum | None = None if curriculum is None else get_curriculum(curriculum)
        if max_episode_steps is None:
            max_episode_steps = 1000 if self.curriculum is None else self.curriculum.horizon
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._recover_steps = None if self.curriculum is None else self.curriculum.recover_steps
        self.sim = HumanoidSim(physics_dt=PHYSICS_DT if curriculum in ("getup", "impact") else None,
                               action_mapping=ACTION_MAPPING if curriculum in ("getup", "impact") else "legacy")
        profile = None if self.curriculum is None else self.curriculum.strikes
        self._schedule = None if profile is None else StrikeSchedule(profile)
        self._progress = 0.0
        self._human_recovery = self.curriculum is not None and self.curriculum.name in ("getup", "impact")
        self._recovery = RecoveryGoal()
        self._history = ObservationHistory()
        self._pose_curriculum = GetupCurriculum() if humanup_stage == "discovery" else DiscoveryPoseCurriculum()
        self._trajectory = DiscoveryTrajectory(reference_path) if self._human_recovery and humanup_stage == "refinement" else None
        if self._trajectory is not None and not explicit_horizon:
            duration = max(self._trajectory.duration(p) for p in ("prone", "supine"))
            self.max_episode_steps = max(self.max_episode_steps, int(np.ceil((duration + 5.1) * self.sim.control_hz)))
        self._tracking_time = 0.
        self._tracking_pose = "supine"
        self._tracking_active = False
        import mujoco
        self._head_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "head")
        self._stand_base_height = float(self.sim.data.xpos[self.sim.pelvis_id, 2])
        self._stand_head_height = float(self.sim.data.geom_xpos[self._head_id, 2])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size(self.sim) if self._human_recovery else self.sim.obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.sim.n_act,), dtype=np.float32)
        self._steps = 0
        self._down_steps = 0
        self._keep_state = False
        self._was_hit = False
        self._knocked_down = False
        self._stable_steps = 0
        self._shots = 0
        self._shot: dict | None = None
        self._renderer = None
        self._assist_fraction = 0.0
        self._assist_n = 0.0
        if self._human_recovery:
            self._reward = (GetupReward(self.sim.weight, control_dt=1. / self.sim.control_hz) if humanup_stage == "discovery"
                            else HumanUPReward(self.sim.weight, humanup_stage, self._trajectory, control_dt=1. / self.sim.control_hz))
        else:
            self._reward = StandReward(self.sim.weight)

    @property
    def assist_fraction(self) -> float:
        return self._assist_fraction

    @assist_fraction.setter
    def assist_fraction(self, value: float) -> None:
        self._assist_fraction = max(0.0, float(value))
        self._assist_n = self._assist_fraction * self.sim.weight

    @property
    def curriculum_progress(self) -> float:
        return self._progress

    @curriculum_progress.setter
    def curriculum_progress(self, value: float) -> None:
        self._progress = float(np.clip(value, 0., 1.))
        if self._schedule is not None:
            self._schedule.progress = self._progress

    def set_curriculum_progress(self, value: float) -> None:
        """VecEnv calls this through Monitor so property setters reach the real env."""
        self.curriculum_progress = value

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._keep_state and seed is None and not options:
            self._keep_state = False
            self._down_steps = 0
            self._steps = 0
            self._reward.reset()
            if self._human_recovery:
                self._history.reset(self.sim)
            observation = self._observation()
            info = self._info(0.0, self.sim.foot_contact)
            info["feet_now"] = info["foot_contact"]
            info["strike"] = None
            return observation, info
        self._keep_state = False
        self._recovery.reset()
        self._was_hit = False
        self._knocked_down = False
        self._stable_steps = 0
        self._shots = 0
        self._shot = None
        self._explicit_pose = bool(options and "pose" in options)
        pose = self._sample_pose()
        if options and "pose" in options:
            pose = options["pose"]
        self.sim.reset(pose, self.np_random if pose == "random" else None)
        if self._human_recovery:
            self._tracking_pose = pose if pose in ("prone", "supine") else "supine"
            self._tracking_time = 0.
            self._tracking_active = pose in ("prone", "supine", "random")
            if self._trajectory is not None and pose == "stand":
                self._tracking_time = self._trajectory.duration(self._tracking_pose)
            # Stage II perturbs initial joint configurations; no mocap reset.
            if self._trajectory is not None and pose in ("prone", "supine") and not options:
                import mujoco
                q = self.sim.data.qpos[self.sim.qpos_adr] + self.np_random.uniform(-.15, .15, self.sim.n_act)
                self.sim.data.qpos[self.sim.qpos_adr] = np.clip(q, self.sim.q_low, self.sim.q_high)
                mujoco.mj_forward(self.sim.model, self.sim.data)
                self.sim._plant_on_ground()
            self._history.reset(self.sim)
        self._reward.reset()
        self._down_steps = 0
        if self._schedule is not None:
            self._schedule.reset(self.np_random)
        if self.curriculum is not None and self.curriculum.name == "impact":
            self._fire_impact_ball()
        self._steps = 0
        observation = self._observation()
        info = self._info(0.0, self.sim.foot_contact)
        info["feet_now"] = info["foot_contact"]
        info["strike"] = self._shot
        if isinstance(self._reward, (HumanUPReward, GetupReward)):
            self._reward.prime(info)
        return observation, info

    def step(self, action):
        if self._human_recovery and self.humanup_stage == "discovery":
            # Taper to zero before standing. Final five seconds are always
            # unassisted, and explicitly requested test poses never get help.
            height = float(self.sim.data.xpos[self.sim.pelvis_id, 2])
            fade = float(np.clip((.75 * self._stand_base_height - height) / (.6 * self._stand_base_height), 0., 1.))
            allowed = (self.curriculum.name == "getup" and not self._explicit_pose
                       and self.sim.pose != "stand" and self.max_episode_steps is not None
                       and self._steps < self.max_episode_steps - 250)
            self.assist_fraction = self._pose_curriculum.assistance * fade if allowed else 0.
        self.sim.set_assist(self._assist_n, upright_only=not isinstance(self._reward, GetupReward))
        self.sim.apply_action(np.asarray(action, dtype=np.float64))
        strike = self._maybe_strike()
        peak_hit = 0.0
        feet = np.zeros(2, dtype=bool)
        for _ in range(self.sim.decimation):
            self.sim.physics_step()
            peak_hit = max(peak_hit, self.sim.last_hit_force)
            feet |= self.sim.foot_contact
            if self.sim.diverged:
                break
        if self._human_recovery:
            self._history.advance(self.sim)
            if self._trajectory is not None:
                state = self.sim.control_state()
                fallen = state["base_height"] < .6 or state["upright"] < .5
                if self.curriculum.name == "impact" and fallen and not self._tracking_active:
                    forward_z = self.sim.data.xmat[self.sim.pelvis_id].reshape(3, 3)[2, 0]
                    self._tracking_pose = "supine" if forward_z > 0. else "prone"
                    self._tracking_time, self._tracking_active = 0., True
                else:
                    self._tracking_time += 1. / self.sim.control_hz
                if not fallen and self._tracking_time >= self._trajectory.duration(self._tracking_pose):
                    self._tracking_active = False
        observation = self._observation()
        info = self._info(peak_hit, feet)
        info["feet_now"] = (bool(self.sim.foot_contact[0]), bool(self.sim.foot_contact[1]))
        info["strike"] = _strike_record(strike)
        if self._human_recovery:
            info["reward_failure"] = bool(self.sim.diverged or not np.isfinite(observation).all())
        reward, terms = self._reward.update(np.asarray(action, dtype=np.float32), info)
        terms["assist"] = float(self._assist_n / self.sim.weight)
        info["reward_terms"] = terms
        self._steps += 1
        standing = _is_standing(info)
        if standing:
            self._down_steps = 0
        else:
            self._down_steps += 1
        recover_failed = self._recover_steps is not None and self._down_steps >= self._recover_steps
        timed_out = self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        torso = float(info["torso_floor_n"]) > TORSO_FORCE and (
            self.curriculum is None or self.curriculum.end_on_torso
        )
        terminated = bool(self.sim.diverged or torso or not np.isfinite(observation).all())
        impact = self.curriculum is not None and self.curriculum.name == "impact"
        if impact and not terminated:
            self._update_impact(info)
        elif self._human_recovery and not terminated:
            self._recovery.update(info, eligible=self._assist_n == 0.)
        if self._human_recovery:
            info["recovery_stage"] = self._recovery.stage
            info["recovery_stable_steps"] = self._recovery.stable_steps
            info["is_recovered"] = self._recovery.complete
            if timed_out and not terminated:
                # A fixed collection horizon is a timeout, not an absorbing
                # failure. SB3 must bootstrap V(terminal_observation), as in
                # upstream HumanUP time_outs handling. Success is separate.
                info["is_success"] = bool(self._recovery.complete and self._assist_n == 0.)
        truncated = bool(timed_out and self._recover_steps is None and not terminated)
        if self._recover_steps is not None and not terminated and (recover_failed or timed_out):
            truncated = True
            info["is_success"] = bool(timed_out and standing and not recover_failed)
        if terminated:
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            info["diverged"] = bool(self.sim.diverged)
            if self._recover_steps is not None and "is_success" not in info:
                info["is_success"] = False
            if self._human_recovery and "is_success" not in info:
                info["is_success"] = False
        if self._human_recovery:
            info["prone_probability"] = self._pose_curriculum.prone_probability
            info["assist_fraction"] = self._assist_fraction
            info["assistance_level"] = getattr(self._pose_curriculum, "level", 3)
            if terminated or truncated:
                self._reward.end_episode(info["base_height"], info["stand_base_height"])
                if not self._explicit_pose and self.curriculum.name == "getup" and self.humanup_stage == "discovery":
                    self._pose_curriculum.update(self.sim.pose, info.get("is_success", False))
        return observation, reward, terminated, truncated, info

    def fire_projectile(self, origin, direction, speed: float | None = None) -> bool:
        if speed is None:
            return self.sim.spawn_projectile(origin, direction)
        return self.sim.spawn_projectile(origin, direction, speed)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.sim.model, camera)
        camera.lookat[:] = self.sim.data.xpos[self.sim.pelvis_id]
        camera.distance = 3.4
        camera.azimuth = 140
        camera.elevation = -15
        self._renderer.update_scene(self.sim.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def _sample_pose(self) -> str:
        if self.curriculum is None:
            return "stand"
        if self.curriculum.name == "getup" and self.humanup_stage == "discovery":
            return self._pose_curriculum.sample(self.np_random)
        names = [name for name, _weight in self.curriculum.poses]
        weights = [weight for _name, weight in self.curriculum.poses]
        return str(self.np_random.choice(names, p=weights))

    def _observation(self):
        if self._human_recovery:
            return self._history.observation(self.sim)
        return self.sim.observation()

    def _fire_impact_ball(self) -> None:
        if self.curriculum is None or self.curriculum.strikes is None:
            raise RuntimeError("impact 没有击球配置")
        profile = self.curriculum.strikes
        low, high = self._schedule._momentum_range()
        momentum = float(self.np_random.uniform(low, high))
        speed = float(profile.speed_cap)
        mass = momentum / speed
        chest = self.sim.data.geom_xpos[self.sim.chest_geom_id]
        origin, direction = impact_ray(self.np_random, chest)
        if not self.sim.spawn_projectile(origin, direction, speed, mass):
            raise RuntimeError("impact 的球没有发出去")
        self._shots += 1
        self._shot = {
            "momentum": momentum,
            "mass": mass,
            "speed": speed,
            "launch": False,
            "patient": False,
        }

    def _update_impact(self, info: dict) -> str | None:
        if float(info["hit_force"]) > 1.0:
            self._was_hit = True
        if self._was_hit and not _is_standing(info):
            self._knocked_down = True  # Diagnostic only; never required for success.
        self._recovery.update(info, eligible=self._was_hit)
        self._stable_steps = self._recovery.stable_steps
        return "success" if self._recovery.complete else None

    def _maybe_strike(self) -> Strike | None:
        if self._schedule is None:
            return None
        if self.curriculum is not None and self.curriculum.name == "impact":
            return None  # One initial ball only; standing must not trigger extra shots.
        info = self.sim.control_state()
        strike = self._schedule.poll(
            info,
            self.sim.data.xpos[self.sim.pelvis_id],
            self.sim.data.geom_xpos[self.sim.chest_geom_id],
        )
        if strike is None:
            return None
        fired = self.sim.spawn_projectile(
            strike.origin,
            strike.direction,
            strike.speed,
            strike.mass,
            strike.gap,
        )
        if not fired:
            return None
        return strike

    def _info(self, hit_force: float, feet: np.ndarray) -> dict:
        info = self.sim.control_state()
        info["hit_force"] = float(hit_force)
        info["foot_contact"] = (bool(feet[0]), bool(feet[1]))
        if self._human_recovery:
            info["chest_upright"] = float(self.sim.data.geom_xmat[self.sim.chest_geom_id].reshape(3, 3)[2, 2])
            info["head_height"] = float(self.sim.data.geom_xpos[self._head_id, 2])
            info["stand_base_height"] = self._stand_base_height
            info["stand_head_height"] = self._stand_head_height
            feet = self.sim.data.xpos[self.sim.foot_body_ids]
            info["feet_height"] = feet[:, 2].copy()
            info["feet_distance"] = float(np.linalg.norm(feet[0, :2] - feet[1, :2]))
            matrices = self.sim.data.xmat[self.sim.foot_body_ids].reshape(-1, 3, 3)
            info["feet_tilt"] = float(np.linalg.norm(matrices[:, 2, :2], axis=1).sum())
            info["joint_q"] = self.sim.data.qpos[self.sim.qpos_adr].copy()
            info["joint_velocity"] = self.sim.data.qvel[self.sim.dof_adr].copy()
            mid = (self.sim.q_low + self.sim.q_high) / 2.
            half = (self.sim.q_high - self.sim.q_low) * .95 / 2.
            info["joint_limit_excess"] = np.maximum(mid - half - info["joint_q"], 0.) + np.maximum(info["joint_q"] - mid - half, 0.)
            info["torque_limit_excess"] = np.maximum(np.abs(info["torque"]) / self.sim.torque_limit - .95, 0.)
            # Reward-only measurements at the final control-step state, matching
            # upstream net rigid-body contacts rather than peak floor sensors.
            import mujoco
            force = np.zeros((2, 3))
            contact_force = np.zeros(6)
            for index in range(self.sim.data.ncon):
                contact = self.sim.data.contact[index]
                bodies = self.sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if not any(body in self.sim.foot_body_ids for body in bodies):
                    continue
                mujoco.mj_contactForce(self.sim.model, self.sim.data, index, contact_force)
                world_force = contact.frame.reshape(3, 3).T @ contact_force[:3]
                for side, body in enumerate(self.sim.foot_body_ids):
                    force[side] += world_force * (int(bodies[1] == body) - int(bodies[0] == body))
            info["feet_force"] = force
            info["feet_speed"] = np.array([self.sim._horizontal_speed(int(body)) for body in self.sim.foot_body_ids])
            if self._trajectory is not None:
                info["tracking_pose"] = self._tracking_pose
                info["tracking_time"] = self._tracking_time
        return info


def _strike_record(strike: Strike | None) -> dict | None:
    if strike is None:
        return None
    return {
        "momentum": float(strike.momentum),
        "mass": float(strike.mass),
        "speed": float(strike.speed),
        "launch": bool(strike.launch),
        "patient": bool(strike.patient),
    }
