"""Gymnasium 环境。站立课用 `StandReward`，getup-plus 把发现奖励和站立奖励叠在一起。

不选课程时，只有模拟发散或骨盆、胸口碰到地面才 `terminated`，超时是 `truncated`。
选了课程之后，倒地超过 8 秒也按失败结束；站着走到时间上限算成功。
getup-human 和 impact 使用参考动作奖励，严格恢复 2 秒再保持 3 秒；固定局长结束时只记评估成败，不加局末奖惩。
impact 每局从站立开场打出一球，扛住球也可算评估成功。局末重置姿势。

动作 [-1, 1]：0 保持 `robot.JOINTS` 里的站立角，±1 顶到关节限位。
观测布局见 `robot.describe_observation()`。
`curriculum` 见 `curriculum.CURRICULA`。查看器不选课程，也不自动发炮弹。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_rl.curriculum import Curriculum, get_curriculum
from robot_rl.rewards import (
    GetupPlusReward,
    DeepMimicReward,
    TORSO_FORCE,
    StandReward,
    _is_standing,
)
from robot_rl.sim import HumanoidSim
from robot_rl.motion import MotionClock, physical_observation, physical_observation_size, sample_reference, tracking_features
from robot_rl.recovery import RecoveryGoal
from robot_rl.strikes import Strike, StrikeSchedule, impact_ray


class HumanoidBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
        curriculum: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("交互观看请运行 python -m robot_rl。这里的 render_mode 只用 rgb_array。")
        self.render_mode = render_mode
        self.curriculum: Curriculum | None = None if curriculum is None else get_curriculum(curriculum)
        if max_episode_steps is None:
            max_episode_steps = 1000 if self.curriculum is None else self.curriculum.horizon
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self._recover_steps = None if self.curriculum is None else self.curriculum.recover_steps
        self.sim = HumanoidSim()
        profile = None if self.curriculum is None else self.curriculum.strikes
        self._schedule = None if profile is None else StrikeSchedule(profile)
        self._progress = 0.0
        self._human_recovery = self.curriculum is not None and self.curriculum.name in ("getup-human", "impact")
        self._recovery = RecoveryGoal()
        self._motion = MotionClock()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(physical_observation_size(self.sim) if self._human_recovery else self.sim.obs_size,),
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
            self._reward = DeepMimicReward(self.sim.weight)
        elif self.curriculum is not None and self.curriculum.name == "getup-plus":
            self._reward = GetupPlusReward(self.sim.weight)
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
        pose = self._sample_pose()
        if options and "pose" in options:
            pose = options["pose"]
        self.sim.reset(pose, self.np_random if pose == "random" else None)
        if self._human_recovery:
            impact = self.curriculum.name == "impact"
            clip = pose if pose in ("prone", "supine") else "prone"
            self._motion.reset(clip)
            if impact or pose == "stand":
                self._motion.time = self._motion.duration
                self._motion.recovering = False
            elif pose == "random":
                self._motion.recovering = False
                self._motion.observe_fall(self.sim)
            # Uniform reference-state initialization during training. Explicit
            # pose resets used by fixed evaluation always start at clip time 0.
            if self.curriculum.name == "getup-human" and not options:
                self._motion.reset(str(self.np_random.choice(("prone", "supine"))))
                self._motion.time = float(self.np_random.uniform(0., self._motion.duration))
                self._set_reference_state()
            elif not impact and pose in ("prone", "supine"):
                self._set_reference_state()
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
        return observation, info

    def step(self, action):
        self.sim.set_assist(self._assist_n)
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
            self._motion.advance(1. / self.sim.control_hz)
            if self.curriculum.name == "impact":
                self._motion.observe_fall(self.sim)
        observation = self._observation()
        info = self._info(peak_hit, feet)
        info["feet_now"] = (bool(self.sim.foot_contact[0]), bool(self.sim.foot_contact[1]))
        info["strike"] = _strike_record(strike)
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
        # Ground contact is expected during get-up. Only apply the paper's
        # torso-contact termination once the reference has reached standing.
        # Impact keeps its recovery window; it must allow a fall after a hit.
        if (self._human_recovery and self.curriculum.name == "getup-human"
                and self._motion.phase >= 1. and info["torso_floor_n"] > TORSO_FORCE):
            terminated = True
        impact = self.curriculum is not None and self.curriculum.name == "impact"
        if impact and not terminated:
            self._update_impact(info)
        elif self._human_recovery and not terminated:
            self._recovery.update(info)
        if self._human_recovery:
            info["recovery_stage"] = self._recovery.stage
            info["recovery_stable_steps"] = self._recovery.stable_steps
            info["is_recovered"] = self._recovery.complete
            if timed_out and not terminated:
                # 固定局长结束。成败只记 is_success，不加局末奖惩。
                terminated = True
                info["is_success"] = self._recovery.complete
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
        names = [name for name, _weight in self.curriculum.poses]
        weights = [weight for _name, weight in self.curriculum.poses]
        return str(self.np_random.choice(names, p=weights))

    def _observation(self):
        if self._human_recovery:
            return physical_observation(self.sim)
        return self.sim.observation()

    def _set_reference_state(self):
        import mujoco
        target = sample_reference(self._motion.clip, self._motion.time)
        root, dof = self.sim.root_qpos_adr, self.sim.root_dof_adr
        self.sim.data.qpos[root:root + 7] = target["qpos"][:7]
        self.sim.data.qpos[self.sim.qpos_adr] = target["q"]
        self.sim.data.qvel[dof:dof + 6] = target["qvel"][:6]
        self.sim.data.qvel[self.sim.dof_adr] = target["qvel"][6:]
        self.sim.q_target[:] = target["q"]
        mujoco.mj_forward(self.sim.model, self.sim.data)

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
            info["reference_clip"] = self._motion.clip
            info["reference_time"] = self._motion.time
            info["reference_phase"] = self._motion.phase
            info["chest_upright"] = float(self.sim.data.geom_xmat[self.sim.chest_geom_id].reshape(3, 3)[2, 2])
            info["motion_features"] = tracking_features(self.sim)
            info["joint_q"] = self.sim.data.qpos[self.sim.qpos_adr].copy()
            info["joint_velocity"] = self.sim.data.qvel[self.sim.dof_adr].copy()
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
