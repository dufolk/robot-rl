import numpy as np
import pytest

from robot_rl.env import HumanoidBalanceEnv
from robot_rl.motion import reference_library, match_reference, sample_reference, tracking_features, body_observation, physical_observation, physical_observation_slices
from robot_rl.recovery import ACQUIRE_STEPS, HOLD_STEPS, RecoveryGoal, is_recovered
from robot_rl.rewards import DeepMimicReward


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


def test_reference_is_finite_in_limits_and_exact_matching_scores_one():
    bank = reference_library()
    env = HumanoidBalanceEnv(curriculum="getup-human")
    for pose in ("prone", "supine"):
        q = bank[f"{pose}_q"]
        assert np.isfinite(q).all()
        assert (q >= env.sim.q_low - 1e-6).all()
        assert (q <= env.sim.q_high + 1e-6).all()
        assert np.all(np.diff(bank[f"{pose}_time"]) > 0)
        np.testing.assert_allclose(q[-1], env.sim.default_q, atol=1e-6)
    target = sample_reference("prone", 0.)
    score = match_reference(target["q"], target["velocity"], target["features"], clip="prone", time=0.)
    assert score["imitation"] == pytest.approx(1.)
    assert not bank["prone_q"].flags.writeable
    env.close()


def test_static_reference_pose_stops_matching_as_time_advances():
    target = sample_reference("prone", 0.)
    start = match_reference(target["q"], target["velocity"], target["features"], clip="prone", time=0.)
    end = match_reference(target["q"], target["velocity"], target["features"], clip="prone", time=100.)
    assert end["imitation"] < start["imitation"] - .5


def test_matching_joint_angles_does_not_hide_wrong_root_orientation():
    target = sample_reference("prone", 0.)
    features = target["features"].copy()
    q = features[15:19].copy()
    features[15:19] = [q[1], -q[0], q[3], -q[2]]
    score = match_reference(target["q"], target["velocity"], features, clip="prone", time=0.)
    assert score["reference_pose"] < 1e-6
    assert score["imitation"] < .36


def test_reward_is_only_phase_conditioned_imitation():
    env = HumanoidBalanceEnv(curriculum="getup-human")
    _, state = env.reset(seed=0, options={"pose": "prone"})
    reward = DeepMimicReward(env.sim.weight)
    value, quiet = reward.update(np.zeros(20), state)
    _, changed = reward.update(np.ones(20), state)
    assert set(quiet) == {"imitation", "total"}
    assert changed == quiet
    assert value == pytest.approx(1., abs=2e-5)
    env.close()


def test_reference_clock_advances_even_when_actions_do_not_change():
    env = HumanoidBalanceEnv(curriculum="getup-human")
    obs, info = env.reset(seed=0, options={"pose": "prone"})
    assert obs.shape == (189,)
    assert info["reference_phase"] == 0.
    _, _, _, _, info = env.step(np.zeros(20))
    assert info["reference_time"] == pytest.approx(.02)
    assert info["reference_phase"] > 0.
    assert "regularization_strength" not in info
    env.close()


def test_reference_initialization_and_seed_reproducibility():
    env = HumanoidBalanceEnv(curriculum="getup-human")
    phases = []
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        target = sample_reference(info["reference_clip"], info["reference_time"])
        np.testing.assert_allclose(env.sim.data.qpos[env.sim.qpos_adr], target["q"])
        np.testing.assert_allclose(env.sim.data.qvel[env.sim.dof_adr], target["velocity"])
        np.testing.assert_array_equal(obs, physical_observation(env.sim))
        phases.append(info["reference_phase"])
    first, first_info = env.reset(seed=42)
    again, again_info = env.reset(seed=42)
    np.testing.assert_array_equal(first, again)
    assert first_info["reference_time"] == again_info["reference_time"]
    assert min(phases) < .25 and max(phases) > .75
    env.close()


def test_fixed_horizon_keeps_standing_data_and_refall_invalidates_success():
    env = HumanoidBalanceEnv(curriculum="getup-human", max_episode_steps=400)
    env.reset(seed=0, options={"pose": "stand"})
    for _ in range(399):
        _, _, terminated, truncated, info = env.step(np.zeros(20))
        assert not terminated and not truncated
    assert info["is_recovered"]
    env._recovery.reset()
    # A completed stand earlier in the episode must not mask a final fall.
    env.sim.reset("supine")
    _, _, terminated, truncated, info = env.step(np.zeros(20))
    assert terminated and not truncated
    assert info["is_success"] is False
    env.close()


def test_deepmimic_changes_observations_but_keeps_hinge_pd_actions():
    old = HumanoidBalanceEnv(curriculum="getup-plus")
    new = HumanoidBalanceEnv(curriculum="getup-human")
    assert old.observation_space.shape == (89,)
    assert new.observation_space.shape == (189,)
    assert old.observation_space != new.observation_space
    assert old.action_space == new.action_space
    old.close()
    new.close()


def test_progress_reaches_real_env_through_monitor_in_single_env_training():
    from stable_baselines3.common.env_util import make_vec_env
    vec = make_vec_env(HumanoidBalanceEnv, n_envs=1, env_kwargs={"curriculum": "getup-human"})
    try:
        vec.env_method("set_curriculum_progress", .75)
        assert vec.envs[0].unwrapped.curriculum_progress == .75
    finally:
        vec.close()


def test_body_observation_is_invariant_to_world_yaw_and_translation():
    import mujoco
    env = HumanoidBalanceEnv(curriculum="getup-human")
    env.reset(seed=0, options={"pose": "stand"})
    before = body_observation(env.sim)
    root = env.sim.root_qpos_adr
    env.sim.data.qpos[root:root + 2] += [4., -3.]
    env.sim.data.qpos[root + 3:root + 7] = [np.cos(.6), 0., 0., np.sin(.6)]
    mujoco.mj_forward(env.sim.model, env.sim.data)
    np.testing.assert_allclose(before, body_observation(env.sim), atol=1e-6)
    env.close()


def test_impact_selects_recovery_once_instead_of_restarting_clock_each_step():
    env = HumanoidBalanceEnv(curriculum="impact")
    env.reset(seed=0)
    assert env._motion.phase == 1.
    env.sim.reset("supine")
    env._motion.observe_fall(env.sim)
    assert env._motion.clip == "supine"
    env._motion.advance(.1)
    env._motion.observe_fall(env.sim)
    assert env._motion.time == pytest.approx(.1)
    env.close()


@pytest.mark.parametrize("curriculum", ["getup-human", "impact"])
def test_policy_observation_has_no_reference_or_curriculum_guidance(curriculum):
    env = HumanoidBalanceEnv(curriculum=curriculum)
    try:
        env.reset(seed=0, options={"pose": "stand"})
        before = env._observation().copy()
        env._motion.reset("supine", time=1.7)
        env.curriculum_progress = .9
        env._steps = 987
        env._recovery.stable_steps = 123
        np.testing.assert_array_equal(before, env._observation())
        assert before.shape == (189,)
    finally:
        env.close()


def test_physical_observation_preserves_history_contacts_without_duplicate_blocks():
    env = HumanoidBalanceEnv(curriculum="getup-human")
    try:
        env.reset(seed=0, options={"pose": "stand"})
        slices = physical_observation_slices(env.sim)
        assert set(slices) == {"base_height", "base_orientation", "base_lin_vel",
                               "base_ang_vel", "body_states", "last_action", "pressure"}
        for action in (np.zeros(20), np.linspace(-.05, .05, 20)):
            obs, _, _, _, _ = env.step(action)
            body = body_observation(env.sim)
            states = body[1:].reshape(-1, 13)
            root = np.flatnonzero(env.sim.robot_body_ids == env.sim.pelvis_id)[0]
            np.testing.assert_array_equal(obs[:1], body[:1])
            np.testing.assert_array_equal(obs[1:11], states[root, 3:])
            np.testing.assert_array_equal(obs[slices["body_states"]], np.delete(states, root, axis=0).ravel())
            np.testing.assert_allclose(obs[slices["last_action"]], action, atol=1e-7)
            np.testing.assert_allclose(obs[slices["pressure"]], env.sim.pressure, rtol=1e-6, atol=1e-7)
        # A previous command / contact has independent meaning even when pose
        # and velocity are identical; neither may disappear in a body-only view.
        before = env._observation().copy()
        env.sim.last_action[:] += .1
        env.sim.pressure[:] += .2
        after = env._observation()
        np.testing.assert_array_equal(before[:slices["last_action"].start], after[:slices["last_action"].start])
        assert not np.array_equal(before[slices["last_action"]], after[slices["last_action"]])
        assert not np.array_equal(before[slices["pressure"]], after[slices["pressure"]])
        assert after.shape == (189,)
    finally:
        env.close()
