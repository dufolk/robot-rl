"""Gymnasium 环境。奖励在 `rewards.compute_reward`，现在是 0。

不会因为跌倒结束回合，这样后面可以训爬起。
只有模拟发散才 `terminated`。超时是 `truncated`。

动作 [-1, 1]：0 保持 `robot.JOINTS` 里的站立角，±1 顶到关节限位。
观测布局见 `robot.describe_observation()`。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_rl.rewards import compute_reward
from robot_rl.sim import HumanoidSim


class HumanoidBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 1000) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("交互观看请运行 python -m robot_rl。这里的 render_mode 只用 rgb_array。")
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.sim = HumanoidSim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sim.obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.sim.n_act,), dtype=np.float32)
        self._steps = 0
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        pose = "stand"
        if options and "pose" in options:
            pose = options["pose"]
        self.sim.reset(pose)
        self._steps = 0
        observation = self.sim.observation()
        info = self._info(0.0, self.sim.foot_contact)
        return observation, info

    def step(self, action):
        self.sim.apply_action(np.asarray(action, dtype=np.float64))
        peak_hit = 0.0
        feet = np.zeros(2, dtype=bool)
        for _ in range(self.sim.decimation):
            self.sim.physics_step()
            peak_hit = max(peak_hit, self.sim.last_hit_force)
            feet |= self.sim.foot_contact
            if self.sim.diverged:
                break
        observation = self.sim.observation()
        info = self._info(peak_hit, feet)
        reward = float(compute_reward(observation, np.asarray(action, dtype=np.float32), info))
        self._steps += 1
        terminated = bool(self.sim.diverged or not np.isfinite(observation).all())
        truncated = bool(self._steps >= self.max_episode_steps)
        if terminated:
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            info["diverged"] = True
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

    def _info(self, hit_force: float, feet: np.ndarray) -> dict:
        info = self.sim.diagnostics()
        info["hit_force"] = float(hit_force)
        info["foot_contact"] = (bool(feet[0]), bool(feet[1]))
        return info
