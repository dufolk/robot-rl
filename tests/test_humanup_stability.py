"""Regression checks for failures found in the first HumanUP experiment."""
import copy

import numpy as np
import pytest
import torch

from robot_rl.env import HumanoidBalanceEnv
from robot_rl.humanup import DiscoveryPoseCurriculum, proprioception, initial_action_std, exploration_scale
from robot_rl.humanup_ppo import HumanUPPPO, adaptive_rate
from robot_rl.rewards import HumanUPReward


def test_fixed_input_scales_and_physical_exploration_units():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        env.reset(options={"pose": "supine"})
        assert env.sim.sim_dt == pytest.approx(.001)
        assert env.sim.decimation == 20
        env.sim.data.qvel[env.sim.dof_adr] = np.arange(20) * .3
        raw, scaled = proprioception(env.sim, False), proprioception(env.sim)
        np.testing.assert_allclose(scaled[:3], raw[:3] * .25)
        np.testing.assert_array_equal(scaled[3:25], raw[3:25])
        np.testing.assert_allclose(scaled[25:45], raw[25:45] * .05, atol=1e-7)
        np.testing.assert_array_equal(scaled[45:], raw[45:])
        sigma = initial_action_std(env.sim) * exploration_scale(env.sim)
        assert sigma.min() == pytest.approx(.05)
        assert sigma.max() == pytest.approx(.2)
    finally:
        env.close()


def test_reward_dt_and_control_step_acceleration_not_solver_acceleration():
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, state = env.reset(options={"pose": "supine"})
        outputs = []
        for dt in (.01, .02):
            reward = HumanUPReward(env.sim.weight, control_dt=dt)
            reward.prime(state)
            updated = dict(state, joint_velocity=np.ones(20) * dt,
                           joint_acc=np.ones(20) * 1e9, base_height=state["base_height"] + .1, upright=1., chest_upright=1.)
            _, terms = reward.update(np.zeros(20), updated)
            assert terms["dof_acc"] == pytest.approx(-1e-7 * 20 * dt)
            assert terms["d_height"] > 0.
            outputs.append(terms)
        for key in ("height", "feet_distance", "dof_acc"):
            assert outputs[1][key] == pytest.approx(2 * outputs[0][key])
        assert outputs[1]["d_height"] == pytest.approx(2 * outputs[0]["d_height"])
    finally:
        env.close()


def test_pose_difficulty_requires_actual_supine_success():
    curriculum = DiscoveryPoseCurriculum()
    for _ in range(100):
        curriculum.update("stand", True)
        curriculum.update("supine", False)
    assert curriculum.prone_probability == 0.
    for _ in range(100):
        curriculum.update("supine", True)
    assert curriculum.prone_probability == pytest.approx(.45)
    poses = {curriculum.sample(np.random.default_rng(i)) for i in range(100)}
    assert poses == {"supine", "stand", "prone"}


def test_adaptive_learning_rate_thresholds_and_bounds():
    assert adaptive_rate(2e-4, .02) == pytest.approx(2e-4 / 1.5)
    assert adaptive_rate(2e-4, .002) == pytest.approx(3e-4)
    assert adaptive_rate(2e-4, .008) == 2e-4
    assert adaptive_rate(2e-4, 0.) == 2e-4
    assert adaptive_rate(1e-5, 1.) == 1e-5
    assert adaptive_rate(1e-2, .001) == 1e-2


def test_rejected_update_restores_actor_critic_and_optimizer():
    torch.set_num_threads(1)
    env = HumanoidBalanceEnv(curriculum="stand")
    try:
        model = HumanUPPPO("MlpPolicy", env, n_steps=8, batch_size=8,
                           policy_kwargs={"net_arch": [16]}, device="cpu", learning_rate=2e-4)
        # Give Adam a nonempty state before taking the transaction snapshot.
        loss = sum(p.square().sum() for p in model.policy.parameters())
        loss.backward()
        model.policy.optimizer.step()
        reference = copy.deepcopy(model.policy)
        model.policy.optimizer.zero_grad()
        loss = sum(p.square().sum() for p in model.policy.parameters())
        loss.backward()
        model.policy.optimizer.step()
        assert model._validate_update(reference, .1)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(model.policy.state_dict()[key], value, rtol=0, atol=0)
        expected = reference.optimizer.state_dict()["state"]
        for key, value in model.policy.optimizer.state_dict()["state"].items():
            for field, tensor in value.items():
                torch.testing.assert_close(tensor, expected[key][field], rtol=0, atol=0)
        assert model.adaptive_learning_rate == pytest.approx(2e-4 / 1.5)
        assert model.rejected_updates == 1
        assert model.learning_rate_ceiling == pytest.approx(2e-4 / 1.5)
        assert not model._validate_update(reference, .008)
    finally:
        env.close()


def test_official_regularization_membership_and_height_thresholds():
    from robot_rl.rewards import regularization_scale
    assert regularization_scale(1., .7) == pytest.approx(1.0001)
    assert regularization_scale(1., .05) == pytest.approx(.9999)
    assert regularization_scale(1., .3) == 1.
    env = HumanoidBalanceEnv(curriculum="getup")
    try:
        _, state = env.reset(options={"pose": "supine"})
        state = dict(state, joint_velocity=np.ones(20), torque=np.ones(20),
                     base_ang_vel=np.ones(3), feet_force=np.array([[0., 0., 10.], [0., 0., 10.]]),
                     feet_speed=np.array([.04, .09]), feet_height=np.array([.05, .05]))
        outputs = []
        for strength in (1., 2.):
            reward = HumanUPReward(env.sim.weight)
            reward.regularization_strength = strength
            reward.prime(state)
            outputs.append(reward.update(np.ones(20) * .1, state)[1])
        for key in ("dof_vel", "feet", "feet_height", "orientation", "standing_pose"):
            assert outputs[1][key] == pytest.approx(2 * outputs[0][key])
        for key in ("action_rate", "torque", "ang_vel", "lin_vel", "feet_slide", "energy", "symmetry", "height"):
            assert outputs[1][key] == pytest.approx(outputs[0][key])
        assert outputs[0]["feet_slide"] == pytest.approx(-(.2 + .3) * .02)
        assert outputs[0]["feet"] == pytest.approx(2.5 * .02)
    finally:
        env.close()


@pytest.mark.parametrize("sharded", [False, True])
def test_fixed_horizon_bootstraps_value_in_actual_sb3_rollout(sharded):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from robot_rl.vec import ShardedVecEnv
    torch.set_num_threads(1)
    if sharded:
        vec = ShardedVecEnv(1, "getup", n_workers=1, compact_infos=True,
                            env_kwargs={"max_episode_steps": 2})
    else:
        vec = DummyVecEnv([lambda: HumanoidBalanceEnv(curriculum="getup", max_episode_steps=2)])
    class Capture(BaseCallback):
        def _on_step(self):
            if self.locals["dones"][0]:
                self.raw_reward = self.locals["infos"][0]["reward_terms"]["total"]
                assert self.locals["infos"][0]["TimeLimit.truncated"]
            return True
    try:
        model = PPO("MlpPolicy", vec, n_steps=2, batch_size=2, n_epochs=1,
                    learning_rate=0., gamma=.99, device="cpu", policy_kwargs={"net_arch": [8]})
        model.policy.predict_values = lambda obs: torch.full((obs.shape[0], 1), 10., device=obs.device)
        callback = Capture()
        model.learn(total_timesteps=2, callback=callback)
        assert model.rollout_buffer.rewards[-1, 0] == pytest.approx(callback.raw_reward + 9.9, abs=1e-5)
    finally:
        vec.close()


def test_nonfinite_failure_at_horizon_is_not_bootstrapped(monkeypatch):
    env = HumanoidBalanceEnv(curriculum="getup", max_episode_steps=1)
    try:
        env.reset(options={"pose": "supine"})
        monkeypatch.setattr(env, "_observation", lambda: np.full(384, np.nan))
        obs, reward, terminated, truncated, info = env.step(np.zeros(20))
        assert terminated and not truncated
        assert np.isfinite(obs).all()
        assert reward <= -10.
        assert not info["is_success"]
    finally:
        env.close()
