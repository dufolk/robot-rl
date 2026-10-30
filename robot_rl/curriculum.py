"""独立课程：stand 站立；getup HumanUP 两阶段起身；impact 单球恢复。"""

from __future__ import annotations

from dataclasses import dataclass

from robot_rl.robot import SHOT_MASS, SHOT_SPEED

# 50 Hz。倒地连续 8 秒还没起来，这一局按失败结束。
RECOVER_STEPS = 400
# impact 一整局 30 秒。没在这之前站稳就是失败。
IMPACT_STEPS = 30 * 50
# 最近这么多局的成功率高于阈值才提前停。不满窗口不判断。
SUCCESS_WINDOW = 10_000
SUCCESS_RATE = 0.9999


@dataclass(frozen=True)
class StrikeProfile:
    momentum: tuple[float, float]
    momentum_end: tuple[float, float] | None = None
    launch_fraction: float = 0.0
    launch_fraction_end: float | None = None
    gap: tuple[int, int] = (40, 80)
    speed_cap: float = 22.0
    mass_range: tuple[float, float] = (1.5, 8.0)
    uniform_momentum: bool = False
    max_pending: int = 4
    fixed_speed: bool = False


@dataclass(frozen=True)
class Curriculum:
    name: str
    description: str
    horizon: int | None
    poses: tuple[tuple[str, float], ...]
    strikes: StrikeProfile | None
    recover_steps: int | None = RECOVER_STEPS
    end_on_torso: bool = True


CURRICULA: dict[str, Curriculum] = {
    "stand": Curriculum(
        name="stand",
        description="不发炮弹，站满 10 秒",
        horizon=500,
        poses=(("stand", 1.0),),
        strikes=None,
    ),
    "getup": Curriculum(
        name="getup",
        description="自主起身与静止保持，成功驱动撤除训练辅助",
        horizon=20 * 50,
        poses=(("stand", .1), ("prone", .45), ("supine", .45)),
        strikes=None,
        recover_steps=None,
        end_on_torso=False,
    ),
    "impact": Curriculum(
        name="impact",
        description="站立开场单发球，允许直接扛住；恢复与静止保持奖励，严格恢复并保持，30 秒结束",
        horizon=IMPACT_STEPS,
        poses=(("stand", 1.0),),
        strikes=StrikeProfile(
            momentum=(0.02 * SHOT_MASS * SHOT_SPEED, 0.1 * SHOT_MASS * SHOT_SPEED),
            momentum_end=(SHOT_MASS * SHOT_SPEED, 5.0 * SHOT_MASS * SHOT_SPEED),
            speed_cap=SHOT_SPEED,
            mass_range=(SHOT_MASS, SHOT_MASS),
            uniform_momentum=True,
            max_pending=1,
            fixed_speed=True,
        ),
        recover_steps=None,
        end_on_torso=False,
    ),
}


def get_curriculum(name: str) -> Curriculum:
    try:
        return CURRICULA[name]
    except KeyError:
        known = ", ".join(CURRICULA)
        raise ValueError(f"未知课程 {name}，可选 {known}") from None
