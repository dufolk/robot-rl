"""Read SB3 metadata and load only the native PyTorch actor for inference."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import re
import zipfile

import cloudpickle
import numpy as np


def checkpoint_path(path):
    path = Path(path)
    return path if path.exists() else Path(str(path) + ".zip")


def read_metadata(path, keys):
    """Deserialize selected metadata without constructing PPO or reading weights."""
    with zipfile.ZipFile(checkpoint_path(path)) as archive:
        raw = json.loads(archive.read("data"))
    result = {}
    for key in keys:
        value = raw.get(key)
        if isinstance(value, dict) and ":serialized:" in value:
            value = cloudpickle.loads(base64.b64decode(value[":serialized:"]))
        result[key] = value
    return result


class InferencePolicy:
    """Deterministic, flat-observation SB3 MLP inference using PyTorch weights.

    No optimizer, rollout buffer, critic construction or orthogonal initialization.
    The checkpoint remains the original SB3 zip; there is no converted cache.
    """
    def __init__(self, path):
        import torch
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.torch_layers import FlattenExtractor
        from robot_rl.humanup_policy import RecoveryPolicy, LATENT_MEAN_LIMIT

        torch.set_num_threads(1)
        data = read_metadata(path, ("policy_class", "policy_kwargs", "observation_space",
                                    "action_space", "humanup_contract"))
        kwargs = data["policy_kwargs"] or {}
        self.observation_space = data["observation_space"]
        self.action_space = data["action_space"]
        self.humanup_contract = data["humanup_contract"]
        self.recovery_actions = data["policy_class"] is RecoveryPolicy
        self.latent_mean_limit = LATENT_MEAN_LIMIT
        if (data["policy_class"] not in (ActorCriticPolicy, RecoveryPolicy)
                or kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor
                or len(self.observation_space.shape) != 1
                or len(self.action_space.shape) != 1):
            raise ValueError("Fast inference requires this project's flat-observation MLP policy")
        activation = kwargs.get("activation_fn", torch.nn.Tanh)
        self.squash_output = kwargs.get("squash_output", False) or self.recovery_actions
        with zipfile.ZipFile(checkpoint_path(path)) as archive:
            weights = torch.load(io.BytesIO(archive.read("policy.pth")),
                                 map_location="cpu", weights_only=True)
        indices = sorted(int(match.group(1)) for key in weights
                         if (match := re.fullmatch(r"mlp_extractor\.policy_net\.(\d+)\.weight", key)))
        if indices != list(range(0, len(indices) * 2, 2)):
            raise ValueError("Unsupported actor layer layout")
        if any(key.startswith(("features_extractor.", "pi_features_extractor.")) for key in weights):
            raise ValueError("Fast inference does not support learned feature extractors")
        layers = []
        input_dim = self.observation_space.shape[0]
        for prefix in [f"mlp_extractor.policy_net.{i}" for i in indices] + ["action_net"]:
            weight, bias = weights[f"{prefix}.weight"], weights[f"{prefix}.bias"]
            if weight.shape[1] != input_dim:
                raise ValueError("Actor weight dimensions do not match the observation/network")
            layer = torch.nn.Linear(input_dim, weight.shape[0])
            layer.load_state_dict({"weight": weight, "bias": bias}, strict=True)
            layers.append(layer)
            if prefix != "action_net":
                layers.append(activation())
            input_dim = weight.shape[0]
        if input_dim != self.action_space.shape[0]:
            raise ValueError("Actor output does not match the action space")
        self.network = torch.nn.Sequential(*layers).eval()

    def predict(self, observation, deterministic=True):
        import torch

        if not deterministic:
            raise ValueError("The viewer's inference policy only supports deterministic actions")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim not in (1, 2) or obs.shape[-1] != self.observation_space.shape[0]:
            raise ValueError("Observation shape does not match the saved policy")
        with torch.inference_mode():
            actions = self.network(torch.as_tensor(obs))
            if self.recovery_actions:
                actions = self.latent_mean_limit * torch.tanh(actions / self.latent_mean_limit)
            if self.squash_output:
                actions = torch.tanh(actions)
            actions = actions.numpy()
        if self.squash_output:
            actions = self.action_space.low + .5 * (actions + 1.) * (self.action_space.high - self.action_space.low)
        else:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions, None
