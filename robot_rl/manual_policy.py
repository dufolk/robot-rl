"""Explicit physical-state prototype policy, using the existing MLP widths."""
import torch
from robot_rl.humanup_policy import RecoveryPolicy


class PrototypeActivation(torch.nn.Module):
    def forward(self, value):
        return torch.softmax(value, dim=-1)


class ManualGetupPolicy(RecoveryPolicy):
    def __init__(self, *args, actor_activations=None, **kwargs):
        super().__init__(*args, **kwargs)
        activations = actor_activations or [PrototypeActivation, torch.nn.Identity]
        for index, activation in enumerate(activations):
            self.mlp_extractor.policy_net[2*index+1] = activation()
