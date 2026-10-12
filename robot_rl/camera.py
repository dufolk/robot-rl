"""自由飞行摄像机。WASD 沿水平朝向移动，鼠标改变朝向。"""

from __future__ import annotations

import math

import numpy as np

_PITCH_LIMIT = math.radians(85.0)
_LOOK_SENS = 0.0022


class FlyCamera:
    def __init__(self) -> None:
        self.eye = np.array([3.0, -2.2, 1.55], dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.look_at(np.array([0.0, 0.0, 1.0], dtype=np.float64))

    def look_at(self, target: np.ndarray) -> None:
        direction = np.asarray(target, dtype=np.float64) - self.eye
        horizontal = math.hypot(float(direction[0]), float(direction[1]))
        self.yaw = math.atan2(float(direction[1]), float(direction[0]))
        self.pitch = math.atan2(float(direction[2]), horizontal)
        self.pitch = min(_PITCH_LIMIT, max(-_PITCH_LIMIT, self.pitch))

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)],
            dtype=np.float64,
        )

    def flat_forward(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0], dtype=np.float64)

    def right(self) -> np.ndarray:
        # 面朝 +X 时，右手方向是 -Y。
        return np.array([math.sin(self.yaw), -math.cos(self.yaw), 0.0], dtype=np.float64)

    def look(self, dx: float, dy: float) -> None:
        self.yaw -= dx * _LOOK_SENS
        self.pitch += -dy * _LOOK_SENS
        self.pitch = min(_PITCH_LIMIT, max(-_PITCH_LIMIT, self.pitch))

    def move(self, wish: np.ndarray, speed: float, dt: float) -> None:
        direction = np.zeros(3, dtype=np.float64)
        direction += float(wish[0]) * self.flat_forward()
        direction += float(wish[1]) * self.right()
        direction[2] += float(wish[2])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8 or dt <= 0.0:
            return
        self.eye += direction / norm * speed * dt
        self.eye[2] = max(0.2, float(self.eye[2]))

    def apply(self, cam, distance: float = 2.0) -> None:
        """把眼睛位置和朝向写成 MuJoCo 自由摄像机。

        MuJoCo 把相机放在 lookat 沿方位角的反方向上，所以方位角要顺着视线取。
        """
        import mujoco

        forward = self.forward()
        cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        cam.lookat[:] = self.eye + forward * distance
        cam.distance = distance
        cam.elevation = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
        cam.azimuth = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
