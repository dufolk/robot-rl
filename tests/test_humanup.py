import numpy as np
import pytest

from robot_rl.env import HumanoidBalanceEnv
from robot_rl.motion import physical_observation, physical_observation_slices
from robot_rl.humanup import proprioception, ObservationHistory, DiscoveryTrajectory, FORMAT, MODEL_CONTRACT, normalized_state_observation
from robot_rl.recovery import ACQUIRE_STEPS, HOLD_STEPS, RecoveryGoal, is_recovered
from robot_rl.rewards import HumanUPReward


def good_state():
    return dict(base_height=.95, upright=1., foot_contact=(True, True), feet_now=(True, True),
                floor_n=np.r_[200., 200., np.zeros(13)], base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3))


def test_recovery_requires_acquisition_and_hold_and_loses_completion_on_refall():
    goal = RecoveryGoal()
    state = good_state()
    for _ in range(ACQUIRE_STEPS - 1):
        assert not goal.update(state)
    assert goal.stage == "getup"
    assert not goal.update(state)
    assert goal.stage == "hold"
    for _ in range(HOLD_STEPS - 1):
        assert not goal.update(state)
    assert goal.update(state)
    assert goal.stage == "complete"
    assert not goal.update(dict(state, base_height=.3))
    assert goal.stage == "getup"
    assert goal.stable_steps == 0


@pytest.mark.parametrize("field,value", [("base_height", .6), ("upright", .8),
    ("chest_upright", .8),
    ("base_lin_vel", np.array([.3, 0., 0.])), ("base_ang_vel", np.array([0., 0., .7])),
    ("feet_now", (False, True)), ("floor_n", np.r_[200., 200., 10., np.zeros(12)])])
def test_strict_recovery_rejects_false_successes(field, value):
    assert not is_recovered(dict(good_state(), **{field: value}))


@pytest.fixture
def reference(tmp_path):
    from robot_rl.robot import JOINTS
    q = np.array([j.default for j in JOINTS])
    data = {"format": np.array(FORMAT), "joint_names": np.array([j.name for j in JOINTS])}
    for pose in ("prone", "supine"):
        data[f"{pose}_time"] = np.array([0., 8.])
        data[f"{pose}_q"] = np.stack([q + .05, q])
        data[f"{pose}_head_height"] = np.array([.2, 1.5])
    path = tmp_path / "humanup.npz"
    np.savez_compressed(path, **data)
    return path


def test_only_one_getup_course_and_no_deepmimic_dependency(monkeypatch):
    import robot_rl.motion as motion
    from robot_rl.curriculum import CURRICULA, get_curriculum
    assert list(CURRICULA) == ["stand", "getup", "impact"]
    for old in ("getup-plus", "getup-human"):
        with pytest.raises(ValueError):
            get_curriculum(old)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no mocap")))
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        obs, info = env.reset(seed=0, options={"pose": "supine"})
        assert obs.shape == (384,)
        assert "reference_phase" not in info and "tracking_time" not in info
        assert not hasattr(env, "_motion")
        _, _, _, _, info = env.step(np.zeros(20))
        assert "imitation" not in info["reward_terms"] and "tracking" not in info["reward_terms"]
        assert "height_deficit" in info["reward_terms"] and "progress" in info["reward_terms"]
        assert "symmetry" not in info["reward_terms"]
    finally:
        env.close()


def test_history_preserves_current_189_and_orders_past_without_cross_episode_leak():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        obs, _ = env.reset(seed=1, options={"pose": "stand"})
        initial = proprioception(env.sim).copy()
        np.testing.assert_array_equal(obs[:189], normalized_state_observation(env.sim))
        np.testing.assert_array_equal(obs[189:].reshape(3, 65), np.tile(initial, (3, 1)))
        obs, _, _, _, _ = env.step(np.ones(20) * .01)
        first = proprioception(env.sim).copy()
        np.testing.assert_array_equal(obs[189:].reshape(3, 65)[-1], initial)
        obs, _, _, _, _ = env.step(np.ones(20) * -.01)
        np.testing.assert_array_equal(obs[189:].reshape(3, 65)[-1], first)
        np.testing.assert_array_equal(obs[:189], normalized_state_observation(env.sim))
        obs, _ = env.reset(seed=1, options={"pose": "supine"})
        now = proprioception(env.sim)
        np.testing.assert_array_equal(obs[189:].reshape(3, 65), np.tile(now, (3, 1)))
    finally:
        env.close()


def test_discovery_reward_prefers_standing_and_penalizes_asymmetric_jitter():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        env.reset(options={"pose": "stand"})
        for _ in range(50):
            _, _, _, _, stand = env.step(np.zeros(20))
        _, lie = env.reset(options={"pose": "supine"})
        reward = HumanUPReward(env.sim.weight)
        standing, _ = reward.update(np.zeros(20), stand)
        reward.reset()
        lying, _ = reward.update(np.zeros(20), lie)
        assert standing > lying
        reward.reset()
        reward.update(np.zeros(20), stand)
        _, quiet = reward.update(np.zeros(20), stand)
        _, jitter = reward.update(np.r_[np.zeros(2), np.ones(6), np.zeros(12)], stand)
        assert jitter["action_rate"] < quiet["action_rate"]
        assert jitter["symmetry"] == quiet["symmetry"] == 0.
        reward.reset()
        _, asymmetric = reward.update(np.r_[np.zeros(2), np.ones(6), np.zeros(12)], lie)
        assert asymmetric["symmetry"] < 0.
        reward.reset()
        reward.update(np.zeros(20), lie)
        _, rise = reward.update(np.zeros(20), dict(lie, base_height=lie["base_height"] + .03))
        assert rise["d_height"] >= 0.
    finally:
        env.close()


@pytest.mark.parametrize("curriculum", ["getup", "impact"])
def test_policy_inputs_have_no_stage_reference_clock_or_goal_guidance(curriculum):
    env = HumanoidBalanceEnv(curriculum=curriculum)
    try:
        env.reset(seed=0, options={"pose": "stand"})
        before = env._observation().copy()
        env._tracking_pose, env._tracking_time = "prone", 100.
        env.curriculum_progress = .9
        env._steps = 987
        env._recovery.stable_steps = 123
        np.testing.assert_array_equal(before, env._observation())
        slices = physical_observation_slices(env.sim)
        assert "last_action" in slices and "pressure" in slices
    finally:
        env.close()


def test_refinement_requires_own_trajectory_and_keeps_reward_clock_out_of_inputs(reference):
    with pytest.raises(ValueError, match="requires"):
        HumanoidBalanceEnv(curriculum="getup", humanup_stage="refinement")
    env = HumanoidBalanceEnv(curriculum="getup", humanup_stage="refinement", reference_path=str(reference))
    try:
        obs, info = env.reset(seed=0, options={"pose": "supine"})
        assert info["tracking_time"] == 0.
        before = env._observation().copy()
        env._tracking_time = 5.
        np.testing.assert_array_equal(before, env._observation())
        _, _, _, _, info = env.step(np.zeros(20))
        assert info["tracking_time"] == pytest.approx(5.02)
        assert "tracking" in info["reward_terms"] and "height" not in info["reward_terms"]
        assert obs.shape == (384,)
    finally:
        env.close()


def test_bad_reference_cannot_fall_back_to_mocap(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, q=np.zeros((10, 20)))
    with pytest.raises(ValueError, match="discovery trajectory"):
        DiscoveryTrajectory(path)


def test_refinement_options_reach_all_sharded_workers(reference):
    from robot_rl.vec import ShardedVecEnv
    vec = ShardedVecEnv(2, "getup", n_workers=2,
                        env_kwargs={"humanup_stage": "refinement", "reference_path": str(reference)})
    try:
        assert vec.get_attr("humanup_stage") == ["refinement", "refinement"]
        assert vec.reset().shape == (2, 384)
        _, rewards, _, infos = vec.step(np.zeros((2, 20)))
        assert np.isfinite(rewards).all()
        assert all("tracking" in info["reward_terms"] for info in infos)
    finally:
        vec.close()


def test_fixed_horizon_keeps_standing_and_final_fall_invalidates_success():
    env = HumanoidBalanceEnv(curriculum="getup", max_episode_steps=400)
    try:
        env.reset(seed=0, options={"pose": "stand"})
        for _ in range(399):
            _, _, terminated, truncated, info = env.step(np.zeros(20))
            assert not terminated and not truncated
        assert info["is_recovered"]
        env.sim.reset("supine")
        _, _, terminated, truncated, info = env.step(np.zeros(20))
        assert not terminated and truncated and not info["is_success"]
    finally:
        env.close()


def test_export_failure_does_not_replace_existing_reference(tmp_path, monkeypatch):
    import robot_rl.export_humanup as exporter
    from stable_baselines3 import PPO
    from types import SimpleNamespace
    model = tmp_path / "model.zip"
    model.write_bytes(b"test")
    output = tmp_path / "keep.npz"
    output.write_bytes(b"keep")
    monkeypatch.setattr(PPO, "load", lambda *a, **kw: SimpleNamespace(observation_space=SimpleNamespace(shape=(384,)), humanup_stage="discovery", humanup_contract=MODEL_CONTRACT))
    monkeypatch.setattr(exporter, "discover_rollout", lambda *a: None)
    with pytest.raises(RuntimeError, match="No validated"):
        exporter.export(model, output, attempts=1)
    assert output.read_bytes() == b"keep"


def test_export_to_refinement_round_trip_has_valid_slowed_timing(tmp_path, monkeypatch):
    import robot_rl.export_humanup as exporter
    from stable_baselines3 import PPO
    from types import SimpleNamespace
    from robot_rl.robot import JOINTS
    model = tmp_path / "model.zip"
    model.write_bytes(b"test")
    monkeypatch.setattr(PPO, "load", lambda *a, **kw: SimpleNamespace(observation_space=SimpleNamespace(shape=(384,)), humanup_stage="discovery", humanup_contract=MODEL_CONTRACT))
    q = np.tile([j.default for j in JOINTS], (5, 1))
    monkeypatch.setattr(exporter, "discover_rollout", lambda *a: (q, np.linspace(.2, 1.5, 5), .02))
    path = exporter.export(model, tmp_path / "ref.npz", slowdown=8.)
    bank = DiscoveryTrajectory(path)
    assert bank.duration("prone") == pytest.approx(.64)
    np.testing.assert_allclose(bank.sample("supine", 1.)[0], q[-1])
