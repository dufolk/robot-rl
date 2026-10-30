"""Strict recovery acquisition and holding, shared by getup and impact."""
import numpy as np

ACQUIRE_STEPS = 2 * 50
HOLD_STEPS = 3 * 50
RECOVERY_STEPS = ACQUIRE_STEPS + HOLD_STEPS


def is_recovered(info: dict) -> bool:
    return (float(info["base_height"]) >= .8 and float(info["upright"]) >= .9
            and float(info.get("chest_upright", info["upright"])) >= .9
            and all(info.get("feet_now", info["foot_contact"]))
            and not np.any(np.asarray(info["floor_n"])[2:] > 1.0)
            and np.linalg.norm(info["base_lin_vel"]) < .25
            and np.linalg.norm(info["base_ang_vel"]) < .6)


class RecoveryGoal:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stable_steps = 0

    def update(self, info: dict, eligible: bool = True) -> bool:
        self.stable_steps = self.stable_steps + 1 if eligible and is_recovered(info) else 0
        return self.complete

    @property
    def complete(self) -> bool:
        return self.stable_steps >= RECOVERY_STEPS

    @property
    def stage(self) -> str:
        if self.complete:
            return "complete"
        return "hold" if self.stable_steps >= ACQUIRE_STEPS else "getup"
