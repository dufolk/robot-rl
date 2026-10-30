"""交互查看器：WASD + 鼠标看，左键打炮弹。

不传策略时动作全为 0，机器人用 PD 撑住站立姿态。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from robot_rl.camera import FlyCamera
from robot_rl.policy import Policy
from robot_rl.robot import SHOT_MASS, SHOT_SPEED
from robot_rl.sim import HumanoidSim

_HELP = """两足人形 MuJoCo 平台
  WASD        水平移动摄像机
  鼠标移动    转向（指针已捕获，Tab 释放）
  鼠标左键    发射炮弹
  空格 / C    上升 / 下降
  Shift       加速
  R           重置机器人
  Esc         退出

全零动作保持站立姿态。奖励在 robot_rl/rewards.py，目前是 0。
环境：HumanoidBalanceEnv。跌倒不会结束回合。
"""


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def run(policy: Policy | None = None, *, seconds: float | None = None) -> None:
    _configure_stdio()
    print(_HELP, flush=True)
    sim = HumanoidSim(physics_dt=getattr(policy, "physics_dt", None),
                      action_mapping=getattr(policy, "action_mapping", "legacy"))
    if not glfw.init():
        raise RuntimeError("GLFW 初始化失败，无法打开查看器")
    window = None
    context = None
    try:
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        window = glfw.create_window(1280, 800, "Humanoid RL Platform", None, None)
        if window is None:
            raise RuntimeError("创建窗口失败")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)
        _loop(window, context, sim, policy, seconds)
    finally:
        # 释放渲染上下文时 GL 窗口还在，析构函数才能把显存还回去。
        context = None
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def _loop(window, context, sim: HumanoidSim, policy: Policy | None, seconds: float | None) -> None:
    scene = mujoco.MjvScene(sim.model, maxgeom=20000)
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    camera = mujoco.MjvCamera()
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True

    fly = FlyCamera()
    shot = {"pending": False}
    captured = {"value": True}
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def on_mouse(_window, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and captured["value"]:
            shot["pending"] = True

    glfw.set_mouse_button_callback(window, on_mouse)

    zeros = np.zeros(sim.n_act, dtype=np.float64)
    step_index = 0
    accum = 0.0
    previous = time.perf_counter()
    started = previous
    cursor_ready = False
    last_cursor = glfw.get_cursor_pos(window)
    reset_was_down = False
    tab_was_down = False
    display_hit = 0.0
    display_hit_until = 0.0

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        if seconds is not None and now - started >= seconds:
            break
        frame_dt = min(0.05, now - previous)
        previous = now
        glfw.poll_events()

        tab_down = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        if tab_down and not tab_was_down:
            captured["value"] = not captured["value"]
            mode = glfw.CURSOR_DISABLED if captured["value"] else glfw.CURSOR_NORMAL
            glfw.set_input_mode(window, glfw.CURSOR, mode)
            cursor_ready = False
        tab_was_down = tab_down

        cursor = glfw.get_cursor_pos(window)
        if captured["value"] and cursor_ready:
            dx = cursor[0] - last_cursor[0]
            dy = cursor[1] - last_cursor[1]
            if abs(dx) < 400.0 and abs(dy) < 400.0:
                fly.look(dx, dy)
        last_cursor = cursor
        cursor_ready = True

        fly.move(_wish(window), _speed(window), frame_dt)

        reset_down = glfw.get_key(window, glfw.KEY_R) == glfw.PRESS
        if reset_down and not reset_was_down:
            sim.reset("stand")
            step_index = 0
        reset_was_down = reset_down

        if shot["pending"]:
            shot["pending"] = False
            sim.spawn_projectile(fly.eye, fly.forward(), speed=SHOT_SPEED, mass=SHOT_MASS)

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        accum = min(0.1, accum + frame_dt)
        frame_hit = 0.0
        while accum >= sim.sim_dt:
            if step_index % sim.decimation == 0:
                if sim.diverged:
                    sim.reset(sim.pose)
                observation = sim.observation()
                if policy is None:
                    action = zeros
                elif isinstance(policy, _SavedPolicy) and policy.uses_body_observation:
                    action = policy.act_sim(sim)
                else:
                    action = np.asarray(policy.act(observation), dtype=np.float64)
                sim.apply_action(action)
            sim.physics_step()
            frame_hit = max(frame_hit, sim.last_hit_force)
            step_index += 1
            accum -= sim.sim_dt

        if frame_hit > 1.0:
            display_hit = frame_hit
            display_hit_until = now + 0.6
        elif now > display_hit_until:
            display_hit = 0.0

        fly.apply(camera)
        mujoco.mjv_updateScene(
            sim.model,
            sim.data,
            option,
            perturb,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, int(width), int(height))
        mujoco.mjr_render(viewport, scene, context)
        _overlay(viewport, context, sim, display_hit)
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, "+", context, 0.5, 0.5, 1.0, 1.0, 1.0)
        glfw.swap_buffers(window)


def _wish(window) -> np.ndarray:
    forward = float(glfw.get_key(window, glfw.KEY_W) == glfw.PRESS) - float(glfw.get_key(window, glfw.KEY_S) == glfw.PRESS)
    right = float(glfw.get_key(window, glfw.KEY_D) == glfw.PRESS) - float(glfw.get_key(window, glfw.KEY_A) == glfw.PRESS)
    vertical = float(glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS) - float(glfw.get_key(window, glfw.KEY_C) == glfw.PRESS)
    return np.array([forward, right, vertical], dtype=np.float64)


def _speed(window) -> float:
    shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    return 10.0 if shift else 4.0


def _overlay(viewport, context, sim: HumanoidSim, hit: float) -> None:
    info = sim.diagnostics()
    state = "FALLEN" if info["fallen"] else "UP"
    feet = info["foot_contact"]
    left = "\n".join(
        [
            "WASD move    mouse look    LMB shoot",
            "Space/C up down    Shift fast",
            "R reset    Tab cursor    Esc quit",
        ]
    )
    right = "\n".join(
        [
            f"{state}  h {info['base_height']:.2f}  up {info['upright']:.2f}",
            f"feet R{int(feet[0])} L{int(feet[1])}  hit {hit:.0f} N",
        ]
    )
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        left,
        right,
        context,
    )


class _SavedPolicy:
    def __init__(self, path: Path) -> None:
        from robot_rl.checkpoint import InferencePolicy

        self._model = InferencePolicy(path)
        shape = self._model.observation_space.shape
        if shape not in ((89,), (157,), (189,), (245,), (384,), (839,)):
            raise ValueError(f"不支持观测维度 {shape}。当前策略只接受身体状态；旧 160 维引导输入模型需要重新训练。")
        self.uses_body_observation = shape != (89,)
        self.physics_dt = None
        self.action_mapping = "legacy"
        self._history = None
        self._last_time = None
        if shape in ((384,), (839,)):
            from robot_rl.humanup import ObservationHistory, SCALED_CONTRACTS, PHYSICS_DT, MODEL_CONTRACT, ACTION_MAPPING, JOINT_LIMIT_CONTRACTS
            contract = getattr(self._model, "humanup_contract", None)
            if contract is not None and contract not in SCALED_CONTRACTS:
                raise ValueError(f"Unsupported HumanUP observation contract: {contract}")
            if (shape == (384,)) != (contract == MODEL_CONTRACT):
                raise ValueError("Checkpoint history length does not match its observation contract")
            self._history = ObservationHistory(normalized=contract in SCALED_CONTRACTS,
                                                steps=(shape[0] - 189) // 65)
            self.physics_dt = PHYSICS_DT if contract in SCALED_CONTRACTS else None
            self.action_mapping = ACTION_MAPPING if contract in JOINT_LIMIT_CONTRACTS else "legacy"

    def act_sim(self, sim: HumanoidSim) -> np.ndarray:
        from robot_rl.motion import body_observation, fused_observation, physical_observation
        size = self._model.observation_space.shape[0]
        if self._history is not None:
            now = float(sim.data.time)
            if self._last_time is None or now <= self._last_time:
                self._history.reset(sim)
            else:
                self._history.advance(sim)
            self._last_time = now
            observation = self._history.observation(sim)
        elif size == 189:
            observation = physical_observation(sim)
        elif size == 245:
            observation = fused_observation(sim)
        else:
            observation = body_observation(sim)
        return self.act(observation)

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _state = self._model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="打开查看器，可以加载一个训练好的策略")
    parser.add_argument("--policy", type=Path, default=None, help="stable-baselines3 的 zip 检查点")
    args = parser.parse_args()
    if args.policy is not None:
        print(f"加载策略 {args.policy}")
    run(None if args.policy is None else _SavedPolicy(args.policy))


if __name__ == "__main__":
    main()
