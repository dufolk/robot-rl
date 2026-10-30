"""Bounded PPO actions with matching sampling, likelihood and entropy."""
import math

import torch
from torch.nn import functional as F
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

LATENT_MEAN_LIMIT = 2.
LOG_STD_MIN = -5.
LOG_STD_MAX = -.7


class RecoveryActionDistribution(SquashedDiagGaussianDistribution):
    def proba_distribution(self, mean_actions, log_std):
        # Bound the latent mean smoothly so it cannot disappear into the hard
        # actuator clip. This does not add weights or change observation inputs.
        mean = LATENT_MEAN_LIMIT * torch.tanh(mean_actions / LATENT_MEAN_LIMIT)
        return super().proba_distribution(mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))

    def entropy(self):
        # Entropy of the EXECUTED tanh action, including the Jacobian. A
        # reparameterized sample gives gradients for both mean and variance.
        latent = self.distribution.rsample()
        log_jacobian = 2. * (math.log(2.) - latent - F.softplus(-2. * latent))
        return self.distribution.entropy().sum(-1) + log_jacobian.sum(-1)


class RecoveryPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if kwargs.get("use_sde", False):
            raise ValueError("RecoveryPolicy uses independent bounded Gaussian actions")
        kwargs.pop("squash_output", None)
        super().__init__(*args, squash_output=False, **kwargs)
        # Same MLP, action head and log_std parameters as ActorCriticPolicy.
        self.action_dist = RecoveryActionDistribution(self.action_space.shape[0])
        self._squash_output = True
