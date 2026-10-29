"""课程里的炮弹计划。

有炮弹的课程都要等机器人站稳 0.5 秒再打。倒着的时候只排队，不发射。
打出一发之后重新计稳定时间，所以下一发一定发生在再次站稳之后。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_rl.curriculum import StrikeProfile
from robot_rl.rewards import is_stable

# 50 Hz 下 25 步 = 0.5 秒。
STRIKE_STABLE_STEPS = 25
COOLDOWN_STEPS = 8
MAX_LAUNCH_SPEED = 22.0
LAUNCH_MOMENTUM = (450.0, 900.0)
LAUNCH_MASS = (12.0, 28.0)


@dataclass(frozen=True)
class _Kind:
    launch: bool
    momentum: float


@dataclass(frozen=True)
class Strike:
    origin: np.ndarray
    direction: np.ndarray
    speed: float
    mass: float
    momentum: float
    launch: bool
    patient: bool
    gap: float = 0.08


class StrikeSchedule:
    def __init__(self, profile: StrikeProfile) -> None:
        self.profile = profile
        self.progress = 0.0
        self.reset(np.random.default_rng(0))

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.stable_steps = 0
        self.pending: list[_Kind] = []
        self.gap = int(self.rng.integers(self.profile.gap[0], self.profile.gap[1]))
        self.cooldown = 0

    def poll(self, info: dict, pelvis: np.ndarray, chest: np.ndarray) -> Strike | None:
        if is_stable(info):
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        self._plan()
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if not self.pending or self.stable_steps < STRIKE_STABLE_STEPS:
            return None
        kind = self.pending.pop(0)
        self.stable_steps = 0
        self.cooldown = COOLDOWN_STEPS
        return self._realize(kind, pelvis, chest)

    def _plan(self) -> None:
        if self.gap > 0:
            self.gap -= 1
            return
        if len(self.pending) < self.profile.max_pending:
            self.pending.append(self._roll())
        self.gap = int(self.rng.integers(self.profile.gap[0], self.profile.gap[1]))

    def _roll(self) -> _Kind:
        launch = bool(self.rng.random() < self._launch_fraction())
        low, high = LAUNCH_MOMENTUM if launch else self._momentum_range()
        if self.profile.uniform_momentum and not launch:
            momentum = float(self.rng.uniform(low, high))
        else:
            momentum = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        return _Kind(launch=launch, momentum=momentum)

    def _launch_fraction(self) -> float:
        end = self.profile.launch_fraction if self.profile.launch_fraction_end is None else self.profile.launch_fraction_end
        start = self.profile.launch_fraction
        blend = float(np.clip(self.progress, 0.0, 1.0))
        return (1.0 - blend) * start + blend * end

    def _momentum_range(self) -> tuple[float, float]:
        end = self.profile.momentum if self.profile.momentum_end is None else self.profile.momentum_end
        blend = float(np.clip(self.progress, 0.0, 1.0))
        low = (1.0 - blend) * self.profile.momentum[0] + blend * end[0]
        high = (1.0 - blend) * self.profile.momentum[1] + blend * end[1]
        return low, high

    def _realize(self, kind: _Kind, pelvis: np.ndarray, chest: np.ndarray) -> Strike:
        pelvis = np.asarray(pelvis, dtype=np.float64)
        chest = np.asarray(chest, dtype=np.float64)
        if kind.launch:
            origin, direction = _launch_ray(self.rng, pelvis)
            mass_low, mass_high = LAUNCH_MASS
            speed_cap = MAX_LAUNCH_SPEED
        else:
            origin, direction = _random_ray(self.rng, chest)
            mass_low, mass_high = self.profile.mass_range
            speed_cap = self.profile.speed_cap
        if self.profile.fixed_speed and not kind.launch:
            # 轻球通过减质量实现，保持速度，避免第一次接触后退役的慢球传不出足够冲量。
            speed = speed_cap
            mass = kind.momentum / speed
        else:
            mass, speed = _split_momentum(kind.momentum, self.rng, mass_low, mass_high, speed_cap)
        return Strike(
            origin=origin,
            direction=direction,
            speed=speed,
            mass=mass,
            momentum=mass * speed,
            launch=kind.launch,
            patient=True,
        )


def _split_momentum(
    momentum: float,
    rng: np.random.Generator,
    mass_low: float,
    mass_high: float,
    speed_cap: float,
) -> tuple[float, float]:
    mass = float(rng.uniform(mass_low, mass_high))
    speed = momentum / mass
    if speed > speed_cap:
        speed = speed_cap
        mass = momentum / speed
    return mass, speed


def impact_ray(rng: np.random.Generator, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """水平随机方向，对准胸口。斜向的球会擦到地面或者从头顶过去。"""
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    distance = float(rng.uniform(1.2, 1.8))
    origin = np.asarray(target, dtype=np.float64).reshape(3) - direction * distance
    return origin, direction


def _random_ray(rng: np.random.Generator, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    distance = float(rng.uniform(1.1, 2.2))
    origin = target - direction * distance
    if origin[2] < 0.2:
        origin = origin.copy()
        origin[2] = 0.2
        direction = target - origin
    return origin, direction


def _launch_ray(rng: np.random.Generator, pelvis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 从身前或身后的低处往上打。水平偏一点，撞击不通过质心，腾空时会带旋转。
    forward = float(rng.choice((-1.0, 1.0)))
    side = float(rng.uniform(-0.14, 0.14))
    distance = float(rng.uniform(0.55, 0.85))
    origin = pelvis + np.array([forward * distance, side, 0.0])
    origin = origin.copy()
    origin[2] = 0.18
    aim = pelvis + np.array([forward * 0.02, side * 0.35, 0.0])
    return origin, aim - origin
