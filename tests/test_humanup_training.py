"""Exercise the actual trainer / save / reload path without a long experiment."""
import numpy as np
import pytest


def test_new_policy_trains_saves_reloads_and_drives_viewer(tmp_path, monkeypatch):
    import torch
    import robot_rl.train as training
    from stable_baselines3.common.callbacks import BaseCallback
    from robot_rl.env import HumanoidBalanceEnv
    from robot_rl.play import _SavedPolicy

    class QuietCallback(BaseCallback):
        def _on_step(self):
            return True

    monkeypatch.setattr(training, "_drop_unsupported_gpu_power", lambda: None)
    experiments = []
    def quiet_swan(**kwargs):
        experiments.append(kwargs["experiment_name"])
        return QuietCallback()
    monkeypatch.setattr(training, "SwanLabCallback", quiet_swan)
    monkeypatch.setattr(training, "_RecoveryEvaluation", lambda *a, **kw: QuietCallback())
    monkeypatch.setattr(training.swanlab, "finish", lambda: None)
    monkeypatch.setattr(training, "RUN_DIR", tmp_path)
    path = tmp_path / "humanoid_ppo_getup"
    model = training.train(64, n_envs=1, n_steps=32, curriculum="getup",
                           device="cpu")
    assert experiments == ["getup"]
    assert path.with_suffix(".zip").exists()
    assert model.num_timesteps == 64
    assert model.ent_coef == .001
    assert model.logger.name_to_value["时间/本轮更新秒数"] > 0.
    assert 0. < model.logger.name_to_value["时间/采样阶段占比"] < 1.
    assert model.logger.name_to_value["训练/熵系数"] == .001
    assert np.isfinite(model.logger.name_to_value["训练/加权熵项"])
    assert model.logger.name_to_value["训练/目标角标准差"] > 0.
    assert model.observation_space.shape == (384,)
    assert model.policy.mlp_extractor.policy_net[0].out_features == 256
    assert model.policy.mlp_extractor.policy_net[2].out_features == 256
    assert isinstance(model.policy.mlp_extractor.policy_net[1], torch.nn.ELU)
    assert model.policy.log_std.requires_grad
    assert torch.isfinite(model.policy.log_std).all()
    # Inference needs only body state, with no reference files or motion clock.
    saved = _SavedPolicy(path.with_suffix(".zip"))
    env = HumanoidBalanceEnv(curriculum="getup")
    assert saved.action_mapping == env.sim.action_mapping == "joint_limits"
    try:
        obs, _ = env.reset(seed=0, options={"pose": "stand"})
        action, _ = model.predict(obs, deterministic=True)
        with monkeypatch.context() as no_reference:
            import robot_rl.humanup as motion
            def forbidden_reference(*args, **kwargs):
                raise AssertionError("inference must not load reference guidance")
            no_reference.setattr(motion, "DiscoveryTrajectory", forbidden_reference)
            np.testing.assert_allclose(saved.act_sim(env.sim), action, atol=1e-6)
        env.sim.reset("supine")
        action = saved.act_sim(env.sim)
        assert not hasattr(saved, "_motion")
        assert action.shape == (20,) and np.isfinite(action).all()
    finally:
        env.close()
    resumed = training.train(96, n_envs=1, n_steps=16, curriculum="getup",
                             device="cpu", save_path=path, load=path.with_suffix(".zip"))
    assert resumed.num_timesteps == 96
    assert resumed.policy.log_std.requires_grad
    # Cross-stage continuation uses the same physical/history input contract.
    from robot_rl.humanup import FORMAT
    from robot_rl.robot import JOINTS
    reference = tmp_path / "ref.npz"
    data = {"format": np.array(FORMAT), "joint_names": np.array([j.name for j in JOINTS])}
    for pose in ("prone", "supine"):
        data[f"{pose}_time"] = np.array([0., 8.])
        data[f"{pose}_q"] = np.tile([j.default for j in JOINTS], (2, 1))
        data[f"{pose}_head_height"] = np.array([.2, 1.5])
    np.savez(reference, **data)
    refined = training.train(112, n_envs=1, n_steps=16, curriculum="getup", device="cpu",
                             save_path=tmp_path / "refined", load=path.with_suffix(".zip"),
                             humanup_stage="refinement", reference=reference)
    assert refined.num_timesteps == 112
    assert refined.humanup_stage == "refinement"
    assert refined.observation_space.shape == (384,)


@pytest.mark.parametrize("legacy_dim", [89, 157, 160, 189, 245, 839])
def test_old_model_is_rejected_before_starting_workers(tmp_path, monkeypatch, legacy_dim):
    from stable_baselines3 import PPO
    import robot_rl.train as training
    from robot_rl.env import HumanoidBalanceEnv

    env = HumanoidBalanceEnv(curriculum="stand")
    if legacy_dim != 89:
        from gymnasium import spaces
        env.observation_space = spaces.Box(-np.inf, np.inf, shape=(legacy_dim,), dtype=np.float32)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu")
    path = tmp_path / "legacy.zip"
    model.save(path)
    env.close()
    monkeypatch.setattr(training, "_drop_unsupported_gpu_power", lambda: None)
    def unexpected_workers(*a, **kw):
        raise AssertionError("legacy validation must happen before workers start")
    monkeypatch.setattr(training, "make_vec_env", unexpected_workers)
    with pytest.raises(ValueError, match="不兼容|不能直接续训"):
        training.train(64, n_envs=1, curriculum="getup", device="cpu",
                       save_path=tmp_path / "new", load=path)
