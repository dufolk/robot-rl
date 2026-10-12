"""策略接口。查看器会在每个控制周期调用 `act`。"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """返回 shape (n_act,) 、范围 [-1, 1] 的关节目标。"""
