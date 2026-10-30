import copy
import math
import os
import time

import numpy as np
import pytest

from robot_rl.camera import FlyCamera
from robot_rl.env import HumanoidBalanceEnv
from robot_rl.robot import DIAGNOSTIC_KEYS, FOOT_CLEARANCE, JOINTS, PRESSURE_SENSORS, describe_observation
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
    assert not any(ball.active for ball in sim._balls)


def test_pressure_sensors_feel_the_ground_and_a_hit():
    from robot_rl.robot import PRESSURE_SENSORS

    sim = HumanoidSim()
    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    _step_for(sim, 0.4)
    pressure = dict(zip(PRESSURE_SENSORS, sim.diagnostics()["pressure"], strict=True))
    assert pressure["right_foot_geom"] > 0.2
    assert pressure["left_foot_geom"] > 0.2
    assert pressure["right_foot_geom"] + pressure["left_foot_geom"] > 0.7
    assert pressure["right_hand"] < 0.05
    assert pressure["head"] < 0.05

    sim.reset("supine")
    sim.apply_action(np.zeros(sim.n_act))
    _step_for(sim, 0.4)
    lying = dict(zip(PRESSURE_SENSORS, sim.diagnostics()["pressure"], strict=True))
    assert lying["chest"] + lying["pelvis"] + lying["head"] > 0.2

    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    import mujoco

    chest_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    chest = sim.data.xpos[chest_id].copy()
    origin = chest + np.array([1.2, 0.0, 0.0])
    assert sim.spawn_projectile(origin, chest - origin, speed=18.0)
    ball = next(item for item in sim._balls if item.active)
    hit = False
    for _ in range(int(0.5 / sim.sim_dt)):
        sim.physics_step()
        if sim.last_hit_force > 20.0:
            hit = True
            break
    assert hit
    assert not ball.active
    felt = dict(zip(PRESSURE_SENSORS, sim.diagnostics()["pressure"], strict=True))
    assert felt["chest"] > 0.02
    _step_for(sim, 0.3)
    assert not ball.active


def test_foot_contact_resists_spin():
    sim = HumanoidSim()
    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    _step_for(sim, 0.2)
    spins = []
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if sim.floor_geom not in pair:
            continue
        if not (pair & set(int(geom_id) for geom_id in sim.foot_geom_ids)):
            continue
        spins.append(float(contact.friction[2]))
    assert spins
    assert min(spins) >= 0.08


def test_origin_stand_is_the_pose_target():
    import mujoco

    from robot_rl.rewards import FEET_SLIDE, StandReward

    sim = HumanoidSim()
    info = sim.diagnostics()
    assert info["body_pos_err"] < 1e-8
    assert info["body_rot_err"] < 1e-8
    home = sim.observation()[sim.obs_slices["home"]]
    np.testing.assert_allclose(home, [0.0, 0.0, 0.0, 1.0], atol=1e-5)

    root = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    sim.data.qpos[sim.model.jnt_qposadr[root]] = 1.0
    mujoco.mj_forward(sim.model, sim.data)
    shifted = sim.diagnostics()
    assert shifted["body_pos_err"] > 0.9
    home = sim.observation()[sim.obs_slices["home"]]
    assert abs(float(home[0])) > 0.5

    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    sim.data.qvel[sim.model.jnt_dofadr[root]] = 2.0
    _step_for(sim, 0.3)
    slipped = sim.diagnostics()
    assert slipped["foot_slip"] > 0.2
    _, terms = StandReward(sim.weight).update(np.zeros(sim.n_act), slipped)
    assert math.isclose(terms["feet_slide"], FEET_SLIDE * slipped["foot_slip"])
    assert terms["termination"] == 0.0


def _reward_state(**overrides):
    info = {
        "base_lin_vel": np.zeros(3),
        "base_ang_vel": np.zeros(3),
        "yaw_rate": 0.0,
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "base_height": 0.95,
        "chest_height": 1.2,
        "foot_contact": (True, True),
        "upright": 1.0,
        "torso_floor_n": 0.0,
        "pelvis_floor_n": 0.0,
        "thigh_floor_n": np.zeros(2),
        "floor_n": np.zeros(len(PRESSURE_SENSORS)),
        "foot_slip": 0.0,
        "joint_pos": np.zeros(4),
        "joint_acc": np.zeros(4),
        "torque": np.zeros(4),
        "diverged": False,
    }
    info.update(overrides)
    return info


def test_action_rate_and_spin_are_penalized():
    from robot_rl.rewards import ACTION_RATE_L2, ANG_VEL_XY_L2, StandReward

    reward = StandReward(400.0)
    _, terms = reward.update(np.zeros(4), _reward_state())
    assert terms["action_rate"] == 0.0
    _, terms = reward.update(np.ones(4), _reward_state())
    assert math.isclose(terms["action_rate"], ACTION_RATE_L2 * 4.0)
    spinning = _reward_state(base_ang_vel=np.array([9.0, 0.0, 0.0]), yaw_rate=3.0)
    _, terms = reward.update(np.ones(4), spinning)
    assert math.isclose(terms["ang_vel_xy"], ANG_VEL_XY_L2 * 81.0)


def test_lateral_push_can_fall():
    sim = HumanoidSim()
    sim.reset("stand")
    sim.apply_action(np.zeros(sim.n_act))
    sim.data.qvel[1] = 3.5
    _step_for(sim, 1.5)
    info = sim.diagnostics()
    assert not info["diverged"]
    assert info["fallen"], info


def test_random_pose_can_fall_from_height():
    sim = HumanoidSim()
    rng = np.random.default_rng(0)
    heights = []
    downward = []
    for _ in range(40):
        sim.reset("random", rng)
        heights.append(sim.lowest_point_z())
        downward.append(float(sim.data.qvel[sim.root_dof_adr + 2]))
        sim.apply_action(np.zeros(sim.n_act))
        _step_for(sim, 0.3)
        assert not sim.diverged
    assert min(heights) < FOOT_CLEARANCE + 0.05
    assert max(heights) > 1.5
    assert any(speed < -0.4 for speed in downward)


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
        assert key in env.sim.diagnostics()
    assert "body_align" not in info
    assert "torso_floor_n" in info
    assert describe_observation(len(JOINTS))
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert "reward_terms" in info2
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


def _floor(**forces: float) -> np.ndarray:
    floor = np.zeros(len(PRESSURE_SENSORS))
    for name, value in forces.items():
        floor[PRESSURE_SENSORS.index(name)] = value
    return floor


def test_reward_events():
    from robot_rl.rewards import UNDESIRED_CONTACTS, StandReward

    reward = StandReward(400.0)
    still, terms = reward.update(np.zeros(4), _reward_state())
    assert math.isclose(terms["lin_vel"], 1.0)
    assert math.isclose(terms["ang_vel"], 1.0)
    assert still > 1.5

    moving = _reward_state(base_lin_vel=np.array([1.0, 0.0, 0.0]))
    _, terms = reward.update(np.zeros(4), moving)
    assert terms["lin_vel"] < 0.05

    _, terms = reward.update(np.zeros(4), _reward_state(floor_n=_floor(right_thigh=2.0)))
    assert math.isclose(terms["undesired"], UNDESIRED_CONTACTS)

    _, terms = reward.update(np.zeros(4), _reward_state(floor_n=_floor(pelvis=2.0)))
    assert math.isclose(terms["undesired"], UNDESIRED_CONTACTS)

    _, terms = reward.update(np.zeros(4), _reward_state(floor_n=_floor(right_shin=2.0, right_hand=2.0)))
    assert math.isclose(terms["undesired"], 2 * UNDESIRED_CONTACTS)

    _, terms = reward.update(np.zeros(4), _reward_state(torso_floor_n=2.0))
    assert terms["termination"] == 0.0

    _, terms = reward.update(np.zeros(4), _reward_state(projected_gravity=np.array([0.0, 0.0, 1.0])))
    assert terms["lin_vel"] == 0.0
    assert terms["ang_vel"] == 0.0


def test_getup_starts_without_assist():
    env = HumanoidBalanceEnv(curriculum="getup")
    assert env._assist_n == 0.0
    env.close()


def test_reward_on_simulated_poses():
    env = HumanoidBalanceEnv()
    env.reset(seed=0)
    info = {}
    reward = 0.0
    for _ in range(5):
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["chest_height"] > 0.8
    assert info["foot_contact"] == (True, True)
    assert reward > 0.0
    assert info["reward_terms"]["lin_vel"] > 0.5
    assert info["reward_terms"]["termination"] == 0.0

    env.reset(options={"pose": "supine"})
    terminated = False
    for _ in range(30):
        _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        if terminated:
            break
    assert info["chest_height"] < 0.8
    assert terminated
    assert info["reward_terms"]["termination"] == 0.0
    env.close()


def test_observation_matches_diagnostics():
    sim = HumanoidSim()
    obs = sim.observation()
    info = sim.diagnostics()
    slc = sim.obs_slices
    np.testing.assert_allclose(obs[slc["base_height"]], [info["base_height"]], atol=1e-5)
    np.testing.assert_allclose(obs[slc["projected_gravity"]], info["projected_gravity"], atol=1e-5)
    assert math.isclose(float(obs[slc["projected_gravity"]][2]), -info["upright"], abs_tol=1e-5)


def test_launch_from_below_lifts_the_robot():
    from robot_rl.strikes import _launch_ray

    sim = HumanoidSim()
    sim.reset("stand")
    action = np.zeros(sim.n_act)
    for _ in range(30):
        sim.apply_action(action)
        for _ in range(sim.decimation):
            sim.physics_step()
    pelvis = sim.data.xpos[sim.pelvis_id].copy()
    origin, direction = _launch_ray(np.random.default_rng(3), pelvis)
    assert sim.spawn_projectile(origin, direction, speed=22.0, mass=32.0, gap=0.05)
    peak_up = 0.0
    feet_off = False
    peak_height = float(pelvis[2])
    for _ in range(50):
        sim.apply_action(action)
        for _ in range(sim.decimation):
            sim.physics_step()
        info = sim.diagnostics()
        peak_up = max(peak_up, info["up_velocity"])
        peak_height = max(peak_height, info["base_height"])
        if not info["foot_contact"][0] and not info["foot_contact"][1]:
            feet_off = True
    assert feet_off or peak_up > 1.0 or peak_height > pelvis[2] + 0.15


def test_impact_fires_one_knockdown_ball_per_episode():
    from robot_rl.curriculum import IMPACT_STEPS, get_curriculum
    from robot_rl.rewards import GetupReward
    from robot_rl.robot import SHOT_MASS, SHOT_SPEED
    from robot_rl.strikes import STRIKE_STABLE_STEPS

    course = get_curriculum("impact")
    assert course.horizon == IMPACT_STEPS
    assert course.poses == (("stand", 1.0),)
    assert course.recover_steps is None
    base = SHOT_MASS * SHOT_SPEED
    assert course.strikes is not None
    assert course.strikes.momentum == pytest.approx((.02 * base, .1 * base))
    assert course.strikes.momentum_end == pytest.approx((base, 5.0 * base))
    assert course.strikes.fixed_speed
    assert course.strikes.speed_cap == SHOT_SPEED

    env = HumanoidBalanceEnv(curriculum="impact")
    assert isinstance(env._reward, GetupReward)
    momenta = []
    for seed in range(12):
        _, info = env.reset(seed=seed)
        assert env.sim.pose == "stand"
        assert env._shots == 1
        assert sum(ball.active for ball in env.sim._balls) == 1
        shot = info["strike"]
        assert shot["speed"] == SHOT_SPEED
        assert .02 * base <= shot["momentum"] <= .1 * base + 1e-8
        assert abs(shot["mass"] * shot["speed"] - shot["momentum"]) < 1e-6
        momenta.append(shot["momentum"])
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
            assert info["strike"] is None
            assert env._shots == 1
            if terminated or truncated:
                break
    env.close()
    assert max(momenta) - min(momenta) > .02 * base
    env = HumanoidBalanceEnv(curriculum="impact")
    env.curriculum_progress = 1.0
    _, info = env.reset(seed=1)
    assert base <= info["strike"]["momentum"] <= 5. * base
    env.close()

    knocked = HumanoidBalanceEnv(curriculum="impact", max_episode_steps=80)
    knocked.curriculum_progress = 1.0
    knocked.reset(seed=2)
    saw_down = False
    for _ in range(40):
        _, _, terminated, truncated, info = knocked.step(np.zeros(knocked.action_space.shape))
        assert not terminated
        assert not truncated
        assert knocked._shots == 1
        if knocked._knocked_down:
            saw_down = True
            break
    assert saw_down
    knocked.close()

    failed = HumanoidBalanceEnv(curriculum="impact", max_episode_steps=30)
    failed.reset(seed=3)
    info = {}
    reward = 0.0
    terminated = False
    truncated = False
    for step in range(1, 31):
        _, reward, terminated, truncated, info = failed.step(np.zeros(failed.action_space.shape))
        if step < 30:
            assert not terminated
            assert not truncated
    assert not terminated
    assert truncated
    assert info["is_success"] is False
    assert "outcome" not in info["reward_terms"]
    assert reward == pytest.approx(info["reward_terms"]["total"])
    qpos = failed.sim.data.qpos.copy()
    failed.reset()
    assert failed.sim.pose == "stand"
    assert failed.sim.data.time == 0.0
    assert not np.allclose(failed.sim.data.qpos, qpos)
    failed.close()

    succeeded = HumanoidBalanceEnv(curriculum="impact", max_episode_steps=400)
    succeeded.reset(seed=4)
    for ball in succeeded.sim._balls:
        succeeded.sim._deactivate(ball)
    for step in range(399):
        if step == 50:
            succeeded._was_hit = True  # Inject a resisted hit after initial foot settling.
        _, _, terminated, _, info = succeeded.step(np.zeros(succeeded.action_space.shape))
        assert not terminated
        assert info["strike"] is None
        assert succeeded._shots == 1
    assert not succeeded._knocked_down
    assert info["is_recovered"]
    _, reward, terminated, truncated, info = succeeded.step(np.zeros(succeeded.action_space.shape))
    assert not terminated
    assert truncated
    assert info["is_success"] is True
    assert "outcome" not in info["reward_terms"]
    assert reward == pytest.approx(info["reward_terms"]["total"])
    qpos = succeeded.sim.data.qpos.copy()
    succeeded.reset()
    assert succeeded.sim.data.time == 0.0
    assert not np.allclose(succeeded.sim.data.qpos, qpos)
    succeeded.close()


def test_mean_policy_kl_is_zero_for_identical_policies_and_grows_when_means_differ():
    import torch
    from robot_rl.train import mean_policy_kl, metric_name

    assert metric_name("train/kl_from_init") == "训练/相对初始KL"
    env = HumanoidBalanceEnv(curriculum="stand")
    try:
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}, "log_std_init": -1.5},
        )
        reference = copy.deepcopy(model.policy)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        obs = np.zeros((16, env.observation_space.shape[0]), dtype=np.float32)
        assert mean_policy_kl(model.policy, reference, obs) == pytest.approx(0.0, abs=1e-5)
        with torch.no_grad():
            model.policy.action_net.bias.add_(0.5)
        assert mean_policy_kl(model.policy, reference, obs) > 0.1
    finally:
        env.close()


def test_logged_metric_names_are_chinese():
    from robot_rl.train import metric_name

    assert metric_name("reward/imitation") == "奖励/模仿"
    assert metric_name("train/value_loss") == "训练/价值损失"
    assert metric_name("rollout/success_rate") == "采样/成功率"
    assert metric_name("eval/recovery_success_rate") == "评估/恢复成功率"


def test_chinese_names_attach_after_logger_exists():
    from types import SimpleNamespace
    from robot_rl.train import _RewardCurves

    recorded = []

    class _Logger:
        def record(self, key, value, exclude=None):
            recorded.append(key)

    curves = _RewardCurves(10)
    curves.init_callback(SimpleNamespace(logger=_Logger()))
    curves._on_training_start()
    curves.logger.record("reward/imitation", 4.0)
    assert recorded == ["奖励/模仿"]


def test_reward_curves_span_more_than_one_episode():
    from types import SimpleNamespace
    from robot_rl.train import _RewardCurves

    recorded = []
    model = SimpleNamespace(logger=SimpleNamespace(
        record=lambda key, value, exclude=None: recorded.append((key, value))))
    curves = _RewardCurves(100)
    curves.init_callback(model)
    quiet = {"reward_terms": {"imitation": 4.0}}
    for _ in range(64):
        curves.locals = {"infos": [quiet]}
        curves._on_step()
    curves._on_rollout_end()
    assert recorded == []
    curves.locals = {"infos": [{"reward_terms": {"imitation": 8.0}}]}
    curves._on_step()
    for _ in range(63):
        curves.locals = {"infos": [quiet]}
        curves._on_step()
    curves._on_rollout_end()
    assert recorded == [("reward/imitation", (8.0 + 4.0 * 127) / 128.0)]


def test_curriculum_progress_uses_only_new_training_steps():
    from types import SimpleNamespace
    from robot_rl.train import _CurriculumProgress

    values = []
    env = SimpleNamespace(env_method=lambda name, value: values.append(("curriculum_progress", value)))
    logger = SimpleNamespace(record=lambda *args: None)
    model = SimpleNamespace(num_timesteps=118_220_544, _total_timesteps=250_000_000,
                            get_env=lambda: env, logger=logger)
    callback = _CurriculumProgress()
    callback.init_callback(model)
    callback._on_training_start()
    callback.num_timesteps = model.num_timesteps
    callback._on_rollout_start()
    assert values[-1] == ("curriculum_progress", 0.0)
    callback.num_timesteps += (250_000_000 - 118_220_544) // 2
    callback._on_rollout_start()
    assert values[-1] == ("curriculum_progress", 0.5)


def test_remaining_courses():
    from robot_rl.curriculum import CURRICULA, get_curriculum

    assert list(CURRICULA) == ["stand", "getup", "impact"]
    assert get_curriculum("stand").strikes is None
    assert get_curriculum("stand").end_on_torso
    human = get_curriculum("getup")
    assert human.horizon == 1000
    assert human.strikes is None
    assert human.recover_steps is None
    assert not human.end_on_torso
    impact = get_curriculum("impact")
    assert impact.poses == (("stand", 1.0),)
    assert impact.horizon == 1500
    assert impact.recover_steps is None
    assert not impact.end_on_torso
    assert impact.strikes is not None
    assert impact.strikes.uniform_momentum
    assert impact.strikes.max_pending == 1


def test_curriculum_success_and_getup_plus():
    from robot_rl.curriculum import RECOVER_STEPS

    stand = HumanoidBalanceEnv(curriculum="stand", max_episode_steps=40)
    stand.reset(seed=0)
    info = {}
    truncated = False
    for _ in range(40):
        _, _, terminated, truncated, info = stand.step(np.zeros(stand.action_space.shape))
        assert info["strike"] is None
        assert not terminated
    assert truncated
    assert info["is_success"] is True
    stand.close()

    fallen = HumanoidBalanceEnv(curriculum="stand", max_episode_steps=500)
    fallen.reset(seed=0, options={"pose": "supine"})
    info = {}
    terminated = False
    for _ in range(30):
        _, _, terminated, _, info = fallen.step(np.zeros(fallen.action_space.shape))
        if terminated:
            break
    assert terminated
    assert info["is_success"] is False
    fallen.close()

    getup = HumanoidBalanceEnv(curriculum="getup", max_episode_steps=RECOVER_STEPS + 50)
    assert type(getup._reward).__name__ == "GetupReward"
    poses = []
    for seed in range(100):
        getup.reset(seed=seed)
        poses.append(getup.sim.pose)
        if getup.sim.pose != "random":
            assert abs(getup.sim.lowest_point_z() - FOOT_CLEARANCE) < 1e-3
    assert set(poses) == {"stand", "supine"}
    assert 15 <= poses.count("stand") <= 45
    assert poses.count("prone") == 0
    assert 55 <= poses.count("supine") <= 85
    getup.reset(seed=0, options={"pose": "supine"})
    info = {}
    terminated = False
    truncated = False
    for _ in range(RECOVER_STEPS + 50):
        _, _, terminated, truncated, info = getup.step(np.zeros(getup.action_space.shape))
        if terminated or truncated:
            break
    assert not terminated and truncated
    assert info["is_success"] is False
    assert "height_deficit" in info["reward_terms"]
    getup.reset(seed=0, options={"pose": "stand"})
    getup.sim.set_assist(120.0)
    getup.sim.physics_step()
    assert getup.sim.data.xfrc_applied[getup.sim.pelvis_id, 2] == 120.0
    getup.reset(seed=0, options={"pose": "supine"})
    getup.sim.set_assist(120.0)
    getup.sim.physics_step()
    assert getup.sim.data.xfrc_applied[getup.sim.pelvis_id, 2] == 0.0
    getup.close()


def test_success_stop_needs_a_full_perfect_window():
    from robot_rl.curriculum import SUCCESS_WINDOW
    from robot_rl.train import ready_to_stop, success_threshold
    assert not ready_to_stop([True] * 20)
    assert ready_to_stop([True] * SUCCESS_WINDOW)
    assert not ready_to_stop([True] * (SUCCESS_WINDOW - 1) + [False])
    assert success_threshold("stand") is not None
    assert success_threshold("getup") is None
    assert success_threshold("impact") is None


def test_prune_checkpoints_keeps_the_latest_per_course(tmp_path):
    from robot_rl.train import prune_checkpoints

    for step in (100, 300, 200):
        (tmp_path / f"humanoid_ppo_stand_{step}_steps.zip").write_bytes(b"x")
    (tmp_path / "humanoid_ppo_50_steps.zip").write_bytes(b"z")
    (tmp_path / "humanoid_ppo_light_10_steps.zip").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    now = time.time()
    os.utime(tmp_path / "humanoid_ppo_stand_300_steps.zip", (now - 30, now - 30))
    os.utime(tmp_path / "humanoid_ppo_stand_200_steps.zip", (now - 20, now - 20))
    os.utime(tmp_path / "humanoid_ppo_stand_100_steps.zip", (now - 10, now - 10))

    assert prune_checkpoints(tmp_path, "humanoid_ppo", keep=1) == 0
    assert prune_checkpoints(tmp_path, "humanoid_ppo_stand", keep=2) == 1
    assert prune_checkpoints(tmp_path / "missing") == 0
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [
        "humanoid_ppo_50_steps.zip",
        "humanoid_ppo_light_10_steps.zip",
        "humanoid_ppo_stand_100_steps.zip",
        "humanoid_ppo_stand_200_steps.zip",
        "notes.txt",
    ]

    assert prune_checkpoints(tmp_path, keep=1) == 1
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [
        "humanoid_ppo_50_steps.zip",
        "humanoid_ppo_light_10_steps.zip",
        "humanoid_ppo_stand_100_steps.zip",
        "notes.txt",
    ]


@pytest.mark.parametrize("curriculum", ["stand", "getup"])
def test_sharded_vec_steps_several_humanoids(curriculum):
    from robot_rl.vec import ShardedVecEnv

    vec = ShardedVecEnv(4, curriculum=curriculum, seed=0, n_workers=2)
    try:
        obs = vec.reset()
        assert obs.shape == (4, vec.observation_space.shape[0])
        vec.set_attr("curriculum_progress", 0.25)
        assert vec.get_attr("curriculum_progress") == [0.25, 0.25, 0.25, 0.25]
        action = np.zeros((4, vec.action_space.shape[0]), dtype=np.float32)
        obs, reward, done, infos = vec.step(action)
        assert obs.shape[0] == 4
        assert reward.shape == (4,)
        assert done.shape == (4,)
        assert len(infos) == 4
        assert "reward_terms" in infos[0]
    finally:
        vec.close()


def test_train_smoke(tmp_path):
    from robot_rl.train import train

    train(timesteps=128, n_envs=1, seed=0, device="cpu", save_path=tmp_path / "ppo", n_steps=64)
    assert (tmp_path / "ppo.zip").is_file()
    logs = list((tmp_path / "swanlog").rglob("*"))
    assert logs
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in logs if path.is_file())
    assert "价值损失" in text
