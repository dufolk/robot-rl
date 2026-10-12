"""平衡 / 爬起奖励。现在固定为 0，训练目标以后改这里。

`info` 的字段见 `robot_rl.robot.DIAGNOSTIC_KEYS`。
`hit_force` 是这一控制周期里炮弹打到身上的最大法向接触力（牛）。
"""

from __future__ import annotations

import numpy as np


def compute_reward(
    observation: np.ndarray,
    action: np.ndarray,
    info: dict,
) -> float:
    del observation, action, info
    return 0.0
