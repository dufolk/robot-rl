import math

import numpy as np
import pytest
import torch

from robot_rl.humanup_policy import RecoveryActionDistribution, RecoveryPolicy
from robot_rl.rewards import HumanUPReward
from robot_rl.env import HumanoidBalanceEnv


def test_target_mapping_covers_each_joint_without_clipped_dead_zones():
    from robot_rl.sim import HumanoidSim
    from robot_rl.humanup import exploration_scale
    sim = HumanoidSim(physics_dt=.001, action_mapping="joint_limits")
    actions = np.linspace(-1., 1., 101)[:, None] * np.ones((1, 20))
    targets = sim.action_targets(actions)
    np.testing.assert_allclose(targets[0], sim.q_low)
    np.testing.assert_allclose(targets[50], sim.default_q)
    np.testing.assert_allclose(targets[-1], sim.q_high)
    assert (np.diff(targets, axis=0) > 0.).all()
    sampled = np.random.default_rng(3).normal(0., .02, (40000, 20))
    np.testing.assert_allclose(sim.action_targets(sampled).std(0), .02 * exploration_scale(sim), rtol=.03)
    legacy = HumanoidSim()
    np.testing.assert_allclose(legacy.action_targets(actions),
        np.clip(legacy.default_q + actions * legacy.action_scale, legacy.q_low, legacy.q_high))


@pytest.mark.parametrize("contract", ["humanup-v2-scaled-dt-1khz", "humanup-v3-bounded-progress"])
def test_old_viewer_keeps_original_history_mapping_and_actions(tmp_path, contract):
    from stable_baselines3 import PPO
    from robot_rl.play import _SavedPolicy
    torch.set_num_threads(1)
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        from gymnasium import spaces
        from robot_rl.humanup import ObservationHistory
        env.observation_space = spaces.Box(-np.inf, np.inf, shape=(839,), dtype=np.float32)
        model = PPO(RecoveryPolicy if "v3" in contract else "MlpPolicy", env, n_steps=8, batch_size=8, device="cpu",
                    policy_kwargs={"net_arch": [16]})
        model.humanup_contract = contract
        path = tmp_path / "old.zip"
        model.save(path)
        saved = _SavedPolicy(path)
        assert saved.action_mapping == ("joint_limits" if "v3" in contract else "legacy")
        assert saved.physics_dt == .001 and saved._history.steps == 10
        env.reset(options={"pose": "supine"})
        history = ObservationHistory(steps=10)
        history.reset(env.sim)
        obs = history.observation(env.sim)
        np.testing.assert_allclose(saved.act_sim(env.sim), model.predict(obs, deterministic=True)[0], atol=1e-6)
    finally:
        env.close()


def test_bounded_actions_have_consistent_likelihood_entropy_and_exploration():
    torch.manual_seed(3)
    raw = torch.full((8192, 2), -1.86, requires_grad=True)
    sigma = torch.tensor([.2, .2]).log().requires_grad_()
    dist = RecoveryActionDistribution(2).proba_distribution(raw, sigma)
    actions = dist.sample()
    assert actions.abs().max() < 1.
    assert actions.std(0).min() > .02
    latent = dist.gaussian_actions
    expected = (dist.distribution.log_prob(latent) - torch.log(1 - actions.square() + dist.epsilon)).sum(-1)
    torch.testing.assert_close(dist.log_prob(actions, latent), expected)
    torch.testing.assert_close(dist.log_prob(actions), expected, rtol=1e-4, atol=2e-4)
    # Unlike entropy of the unclipped Gaussian, executed-action entropy has
    # a gradient for means stuck near the actuator boundaries.
    entropy = dist.entropy().mean()
    assert torch.isfinite(entropy)
    entropy.backward()
    assert raw.grad.abs().sum() > .01
    assert torch.isfinite(sigma.grad).all()


def test_extreme_logits_remain_finite_and_do_not_destroy_exploration():
    raw = torch.full((4096, 20), 100.)
    dist = RecoveryActionDistribution(20).proba_distribution(raw, torch.zeros(20))
    actions = dist.sample()
    assert (actions.abs() < 1.).all()
    assert actions.std(0).min() > .02
    assert torch.isfinite(dist.log_prob(actions)).all()
    assert torch.isfinite(dist.entropy()).all()


def test_squashed_policy_fast_loader_matches_training_and_preserves_architecture(tmp_path):
    from stable_baselines3 import PPO
    from robot_rl.checkpoint import InferencePolicy
    from robot_rl.humanup import MODEL_CONTRACT
    torch.set_num_threads(1)
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        model = PPO(RecoveryPolicy, env, n_steps=8, batch_size=8, device="cpu",
                    policy_kwargs={"net_arch": [32, 16], "activation_fn": torch.nn.ELU})
        model.humanup_contract = MODEL_CONTRACT
        with torch.no_grad():
            model.policy.action_net.bias.copy_(torch.linspace(-5., 5., 20))
        path = tmp_path / "policy.zip"
        model.save(path)
        saved = InferencePolicy(path)
        obs = np.random.default_rng(0).normal(size=(20, 384)).astype(np.float32)
        expected = model.predict(obs, deterministic=True)[0]
        np.testing.assert_allclose(saved.predict(obs)[0], expected, rtol=1e-6, atol=1e-7)
        assert model.policy.action_net.out_features == 20
        assert (np.abs(expected) < 1.).all()
        assert model.policy.log_std.shape == (20,)
    finally:
        env.close()


def recovery_states(env):
    _, stand = env.reset(options={"pose": "stand"})
    floor = np.zeros(15)
    floor[:2] = env.sim.weight / 2.
    stand.update(floor_n=floor, foot_contact=(True, True), feet_now=(True, True),
                 base_lin_vel=np.zeros(3), base_ang_vel=np.zeros(3), upright=1., chest_upright=1.)
    seated_floor = np.zeros(15)
    seated_floor[1], seated_floor[12] = .2 * env.sim.weight, .8 * env.sim.weight
    seated = dict(stand, base_height=.145, head_height=.722, floor_n=seated_floor,
                  foot_contact=(False, True), feet_now=(False, True))
    inverted_floor = floor * .2
    inverted_floor[14] = .8 * env.sim.weight
    inverted = dict(stand, base_height=.622, head_height=.11, upright=-.65, chest_upright=-.94,
                    projected_gravity=np.array([math.sqrt(1 - .65 ** 2), 0., .65]), floor_n=inverted_floor)
    return stand, seated, inverted


def test_discovery_uses_upstream_static_rewards_without_custom_support_gates():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        stand, seated, inverted = recovery_states(env)
        for state in (stand, seated, inverted):
            reward = HumanUPReward(env.sim.weight)
            reward.prime(state)
            _, terms = reward.update(np.zeros(20), state)
            assert terms["height"] == pytest.approx(5 * np.expm1(np.clip(state["base_height"], 0, .728)) * .02)
            assert terms["head_height"] == pytest.approx(5 * np.expm1(np.clip(state["head_height"], 0, 1.3)) * .02)
            assert terms["d_height"] == terms["d_feet"] == 0.
            assert "standing" not in terms and "time_cost" not in terms
    finally:
        env.close()


def test_discovery_binary_increments_and_upstream_termination_weight():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, state = env.reset(options={"pose": "supine"})
        reward = HumanUPReward(env.sim.weight)
        reward.prime(state)
        raised = dict(state, base_height=state["base_height"] + .001,
                      feet_force=np.array([[0., 0., 10.], [0., 0., 10.]]))
        _, first = reward.update(np.zeros(20), raised)
        _, same = reward.update(np.zeros(20), raised)
        assert first["d_height"] == first["d_feet"] == pytest.approx(.02)
        assert same["d_height"] == same["d_feet"] == 0.
        value, terms = reward.update(np.zeros(20), {"diverged": True})
        assert value == pytest.approx(-500 * .02)
        assert set(terms) == {"termination", "total"}
    finally:
        env.close()
