import math

import numpy as np
import pytest

from robot_rl.camera import FlyCamera
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.robot import DIAGNOSTIC_KEYS, FOOT_CLEARANCE, JOINTS, describe_observation
from robot_rl.sim import HumanoidSim


def _step_for(sim: HumanoidSim, duration: float) -> None:
    steps = int(duration / sim.sim_dt)
    for _ in range(steps):
        sim.physics_step()


def test_zero_action_holds_stand():
    sim = HumanoidSim()
    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    assert abs(sim.lowest_point_z() - FOOT_CLEARANCE) < 1e-3

    _step_for(sim, 3.0)
    info = sim.diagnostics()
    assert not info["diverged"], info
    assert info["base_height"] > 0.7, info
    assert info["upright"] > 0.9, info
    assert not info["fallen"], info
    assert info["foot_contact"] == (True, True), info
    drift = np.max(np.abs(sim.data.qpos[sim.qpos_adr] - sim.default_q))
    assert drift < 0.08, drift
    com = info["com"]
    feet = info["foot_pos"]
    assert feet[:, 0].min() - 0.05 < com[0] < feet[:, 0].max() + 0.05
    assert feet[:, 1].min() - 0.05 < com[1] < feet[:, 1].max() + 0.05
    parked = sim.data.qpos[sim._balls[0].qpos_adr + 2]
    assert parked < -5.0


def test_action_zero_is_stand_and_extremes_hit_limits():
    sim = HumanoidSim()
    sim.apply_action(np.zeros(sim.n_act))
    np.testing.assert_allclose(sim.q_target, sim.default_q)
    sim.apply_action(np.ones(sim.n_act))
    np.testing.assert_allclose(sim.q_target, sim.q_high)
    sim.apply_action(-np.ones(sim.n_act))
    np.testing.assert_allclose(sim.q_target, sim.q_low)


def test_projectile_hits_and_lands_where_aimed():
    import mujoco

    sim = HumanoidSim()
    sim.reset("stand")
    chest_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    chest = sim.data.xpos[chest_id].copy()
    origin = chest + np.array([1.5, 0.0, 0.05])
    direction = chest - origin
    assert sim.spawn_projectile(origin, direction, speed=25.0)
    ball = next(item for item in sim._balls if item.active)
    expected = origin + direction / np.linalg.norm(direction) * 0.4
    np.testing.assert_allclose(sim.data.xpos[ball.body_id], expected, atol=1e-4)

    peak = 0.0
    _begin = sim.data.time
    for _ in range(int(0.6 / sim.sim_dt)):
        sim.physics_step()
        peak = max(peak, sim.last_hit_force)
    assert peak > 20.0, peak
    assert not sim.diverged


def test_lateral_push_can_fall():
    sim = HumanoidSim()
    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    sim.data.qvel[1] = 3.5
    _step_for(sim, 1.5)
    info = sim.diagnostics()
    assert not info["diverged"]
    assert info["fallen"], info


def test_supine_starts_fallen_and_stays_finite():
    sim = HumanoidSim()
    sim.reset("supine")
    info = sim.diagnostics()
    assert info["fallen"], info
    assert info["pose"] == "supine"
    _step_for(sim, 1.0)
    assert not sim.diverged
    assert np.isfinite(sim.data.qpos).all()


def test_env_step_contract():
    env = HumanoidBalanceEnv(max_episode_steps=5)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.sim.obs_size,)
    assert obs.dtype == np.float32
    assert env.sim.obs_size == env.observation_space.shape[0]
    for key in DIAGNOSTIC_KEYS:
        assert key in info
    assert describe_observation(len(JOINTS))
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert reward == 0.0
    assert obs2.shape == obs.shape
    assert not terminated
    assert not truncated
    assert "hit_force" in info2
    env.reset(options={"pose": "prone"})
    assert env.sim.pose == "prone"
    truncated = False
    for _ in range(5):
        _, _, _, truncated, _ = env.step(np.zeros(env.sim.n_act))
    assert truncated
    env.close()


def test_camera_tracks_mouse_and_mujoco_view():
    import mujoco

    fly = FlyCamera()
    fly.eye[:] = np.array([0.0, 0.0, 1.0])
    fly.yaw = 0.0
    fly.pitch = 0.0
    np.testing.assert_allclose(fly.forward(), [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(fly.right(), [0.0, -1.0, 0.0], atol=1e-8)

    fly.look(100.0, 0.0)
    assert fly.forward()[1] < 0.0
    fly.yaw = 0.0
    fly.pitch = 0.0
    fly.look(0.0, -80.0)
    assert fly.pitch > 0.0
    assert fly.forward()[2] > 0.0

    fly.eye[:] = np.array([-2.5, -1.2, 1.4])
    fly.yaw = 0.45
    fly.pitch = -0.25
    sim = HumanoidSim()
    cam = mujoco.MjvCamera()
    fly.apply(cam)
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(sim.model, maxgeom=100)
    mujoco.mjv_updateScene(
        sim.model,
        sim.data,
        option,
        perturb,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    forward = np.array(scene.camera[0].forward, dtype=np.float64).reshape(-1)[:3]
    position = np.array(scene.camera[0].pos, dtype=np.float64).reshape(-1)[:3]
    forward /= np.linalg.norm(forward)
    np.testing.assert_allclose(position, fly.eye, atol=5e-2)
    np.testing.assert_allclose(forward, fly.forward(), atol=1e-3)


def test_rgb_render():
    env = HumanoidBalanceEnv(render_mode="rgb_array")
    env.reset()
    try:
        frame = env.render()
    except Exception as exc:
        pytest.skip(f"GL 不可用: {exc}")
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert int(frame.mean()) > 5
    env.close()


def test_observation_matches_diagnostics():
    sim = HumanoidSim()
    obs = sim.observation()
    info = sim.diagnostics()
    slc = sim.obs_slices
    np.testing.assert_allclose(obs[slc["base_height"]], [info["base_height"]], atol=1e-5)
    np.testing.assert_allclose(obs[slc["projected_gravity"]], info["projected_gravity"], atol=1e-5)
    assert math.isclose(float(obs[slc["projected_gravity"]][2]), -info["upright"], abs_tol=1e-5)
