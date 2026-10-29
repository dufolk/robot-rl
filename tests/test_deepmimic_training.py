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
    monkeypatch.setattr(training, "SwanLabCallback", lambda **kw: QuietCallback())
    monkeypatch.setattr(training, "_RecoveryEvaluation", lambda *a, **kw: QuietCallback())
    monkeypatch.setattr(training.swanlab, "finish", lambda: None)
    path = tmp_path / "deepmimic"
    model = training.train(64, n_envs=1, n_steps=32, curriculum="getup-human",
                           device="cpu", save_path=path)
    assert model.num_timesteps == 64
    assert model.observation_space.shape == (189,)
    assert model.policy.mlp_extractor.policy_net[0].out_features == 1024
    assert model.policy.mlp_extractor.policy_net[2].out_features == 512
    assert isinstance(model.policy.mlp_extractor.policy_net[1], torch.nn.ReLU)
    assert not model.policy.log_std.requires_grad
    np.testing.assert_allclose(model.policy.log_std.detach().numpy(), training.LOG_STD_INIT)
    # Inference needs only body state, with no reference files or motion clock.
    saved = _SavedPolicy(path.with_suffix(".zip"))
    env = HumanoidBalanceEnv(curriculum="getup-human")
    try:
        obs, _ = env.reset(seed=0, options={"pose": "stand"})
        action, _ = model.predict(obs, deterministic=True)
        with monkeypatch.context() as no_reference:
            import robot_rl.motion as motion
            def forbidden_reference():
                raise AssertionError("inference must not load reference guidance")
            no_reference.setattr(motion, "reference_library", forbidden_reference)
            np.testing.assert_allclose(saved.act_sim(env.sim), action, atol=1e-6)
        env.sim.reset("supine")
        action = saved.act_sim(env.sim)
        assert not hasattr(saved, "_motion")
        assert action.shape == (20,) and np.isfinite(action).all()
    finally:
        env.close()
    resumed = training.train(96, n_envs=1, n_steps=16, curriculum="getup-human",
                             device="cpu", save_path=path, load=path.with_suffix(".zip"))
    assert resumed.num_timesteps == 96
    assert not resumed.policy.log_std.requires_grad


@pytest.mark.parametrize("legacy_dim", [89, 157, 160, 245])
def test_old_model_is_rejected_before_starting_workers(tmp_path, monkeypatch, legacy_dim):
    from stable_baselines3 import PPO
    import robot_rl.train as training
    from robot_rl.env import HumanoidBalanceEnv

    env = HumanoidBalanceEnv(curriculum="getup-plus")
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
    with pytest.raises(ValueError, match="不兼容"):
        training.train(64, n_envs=1, curriculum="getup-human", device="cpu",
                       save_path=tmp_path / "new", load=path)
