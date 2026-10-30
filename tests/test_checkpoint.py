"""Inference is equivalent to SB3 without loading training-only state."""
import zipfile

import numpy as np
import pytest


@pytest.mark.parametrize("dimension,activation", [(89, "Tanh"), (157, "Tanh"), (189, "Tanh"),
                                                   (245, "Tanh"), (384, "ELU"), (839, "ELU")])
def test_torch_actor_matches_sb3_and_skips_optimizer(tmp_path, monkeypatch, dimension, activation):
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from robot_rl.checkpoint import InferencePolicy, read_metadata

    torch.set_num_threads(1)
    env = gym.make("Pendulum-v1")
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (dimension,), dtype=np.float32)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu",
                policy_kwargs={"net_arch": [32, 16], "activation_fn": getattr(torch.nn, activation)})
    path = tmp_path / "policy.zip"
    model.save(path)
    obs = np.random.default_rng(7).normal(size=(17, dimension)).astype(np.float32)
    expected = model.predict(obs, deterministic=True)[0]
    original_read = zipfile.ZipFile.read
    reads = []

    def read(archive, name, *args, **kwargs):
        reads.append(name)
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    loaded = InferencePolicy(path)
    np.testing.assert_allclose(loaded.predict(obs)[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(loaded.predict(obs[0])[0], expected[0], rtol=1e-6, atol=1e-7)
    assert "policy.pth" in reads
    assert "policy.optimizer.pth" not in reads
    reads.clear()
    metadata = read_metadata(path, ("observation_space", "action_space"))
    assert metadata["observation_space"].shape == (dimension,)
    assert reads == ["data"]
    # The same clipping as SB3 is essential for large actor outputs.
    with torch.no_grad():
        model.policy.action_net.bias.fill_(100.)
    model.save(path)
    loaded = InferencePolicy(path)
    np.testing.assert_array_equal(loaded.predict(obs)[0], model.predict(obs, deterministic=True)[0])
    env.close()


def test_training_cli_does_not_launch_browser_or_dashboard(monkeypatch):
    import sys
    import subprocess
    import webbrowser
    import robot_rl.train as training

    calls = []
    monkeypatch.setattr(sys, "argv", ["robot_rl.train", "--curriculum", "getup", "--timesteps", "123"])
    monkeypatch.setattr(training, "train", lambda **kwargs: calls.append(kwargs))
    def forbidden(*args, **kwargs):
        raise AssertionError("CLI must not open a browser or start a dashboard")
    monkeypatch.setattr(webbrowser, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    training.main()
    assert calls[0]["timesteps"] == 123
