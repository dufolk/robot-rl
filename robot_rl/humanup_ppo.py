"""SB3 PPO with HumanUP's analytic-KL learning-rate schedule.

Upstream: RunpeiDong/HumanUP, simulation/rsl_rl/rsl_rl/algorithms/ppo_rma.py.
This port retains SB3, not ROA. The whole-update rejection is an additional
guard: SB3's early stop alone cannot undo minibatches already applied.
"""
import copy
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO

DESIRED_KL = .008


def adaptive_rate(rate, kl, desired=DESIRED_KL):
    if kl > 2. * desired:
        return max(1e-5, rate / 1.5)
    if 0. < kl < desired / 2.:
        return min(1e-2, rate * 1.5)
    return rate


class HumanUPPPO(PPO):
    def __init__(self, *args, desired_kl=DESIRED_KL, **kwargs):
        self.desired_kl = float(desired_kl)
        self.adaptive_learning_rate = float(kwargs.get("learning_rate", 2e-4))
        self.rejected_updates = 0
        self.learning_rate_ceiling = 1e-2
        self._accepted_streak = 0
        self._diagnostic_updates = 0
        kwargs.setdefault("target_kl", self.desired_kl)
        super().__init__(*args, **kwargs)

    def _set_adaptive_rate(self):
        for group in self.policy.optimizer.param_groups:
            group["lr"] = self.adaptive_learning_rate

    def _update_learning_rate(self, optimizers):
        # The base class otherwise resets a fixed lr at every rollout.
        self._set_adaptive_rate()
        self.logger.record("train/learning_rate", self.adaptive_learning_rate)

    def _rollout_kl(self, reference):
        observations = self.rollout_buffer.observations.reshape(-1, self.observation_space.shape[0])
        total = 0.
        with torch.no_grad():
            for offset in range(0, len(observations), 2048):
                obs = torch.as_tensor(observations[offset:offset + 2048], device=self.device)
                old = reference.get_distribution(obs).distribution
                new = self.policy.get_distribution(obs).distribution
                values = torch.distributions.kl_divergence(old, new).sum(-1)
                total += float(values.sum())
        return total / len(observations)

    def _validate_update(self, reference, kl):
        rejected = not np.isfinite(kl) or kl > 2. * self.desired_kl
        if rejected:
            self.policy.load_state_dict(reference.state_dict())
            self.policy.optimizer.load_state_dict(reference.optimizer.state_dict())
            self.adaptive_learning_rate = max(1e-5, self.adaptive_learning_rate / 1.5)
            # Keep the backoff across minibatches. Otherwise the upstream
            # small-KL growth immediately undoes it on the next rollout.
            self.learning_rate_ceiling = min(self.learning_rate_ceiling, self.adaptive_learning_rate)
            self._accepted_streak = 0
            self.rejected_updates += 1
            self._set_adaptive_rate()
        else:
            self._accepted_streak += 1
            if self._accepted_streak >= 10:
                self.learning_rate_ceiling = min(1e-2, self.learning_rate_ceiling * 1.1)
                self._accepted_streak = 0
        return rejected

    def train(self):
        update_started = perf_counter()
        reference = copy.deepcopy(self.policy)
        reference.set_training_mode(False)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self._diagnostic_updates += 1
        if self._diagnostic_updates % 20 == 1:
            self._record_gradient_balance()
        original = self.policy.evaluate_actions
        had_instance_method = "evaluate_actions" in self.policy.__dict__
        entropies = []

        def evaluate_actions(observations, actions):
            # Upstream adjusts lr using closed-form KL before each minibatch.
            with torch.no_grad():
                old = reference.get_distribution(observations).distribution
                new = self.policy.get_distribution(observations).distribution
                kl = float(torch.distributions.kl_divergence(old, new).sum(-1).mean())
            self.adaptive_learning_rate = min(self.learning_rate_ceiling,
                adaptive_rate(self.adaptive_learning_rate, kl, self.desired_kl))
            self._set_adaptive_rate()
            result = original(observations, actions)
            if result[2] is not None:
                entropies.append(float(result[2].mean().detach()))
            return result

        self.policy.evaluate_actions = evaluate_actions
        try:
            super().train()
        finally:
            if had_instance_method:
                self.policy.evaluate_actions = original
            else:
                del self.policy.evaluate_actions
        kl = self._rollout_kl(reference)
        rejected = self._validate_update(reference, kl)
        self.logger.record("train/proposed_kl", kl)
        self.logger.record("train/analytic_kl", 0. if rejected else kl)
        self.logger.record("train/learning_rate_ceiling", self.learning_rate_ceiling)
        self.logger.record("train/update_rejected", float(rejected))
        self.logger.record("train/rejected_updates", self.rejected_updates)
        self.logger.record("train/learning_rate", self.adaptive_learning_rate)
        self.logger.record("train/entropy_coefficient", self.ent_coef)
        if entropies:
            self.logger.record("train/weighted_entropy_loss", -self.ent_coef * float(np.mean(entropies)))
        if hasattr(self, "humanup_action_scale"):
            scale = torch.as_tensor(self.humanup_action_scale, device=self.device)
            observations = self.rollout_buffer.observations.reshape(-1, self.observation_space.shape[0])[:2048]
            with torch.no_grad():
                distribution = self.policy.get_distribution(torch.as_tensor(observations, device=self.device))
                first, second = distribution.sample().clamp(-1., 1.), distribution.sample().clamp(-1., 1.)
                if hasattr(self, "humanup_action_scale_low"):
                    lower = torch.as_tensor(self.humanup_action_scale_low, device=self.device)
                    upper = torch.as_tensor(self.humanup_action_scale_high, device=self.device)
                    delta = first * torch.where(first >= 0., upper, lower) - second * torch.where(second >= 0., upper, lower)
                else:
                    delta = (first - second) * scale
                effective_std = (.5 * delta.square().mean(0)).sqrt().mean()
                nominal_std = (distribution.distribution.scale * scale).mean()
            self.logger.record("train/target_angle_std", float(effective_std))
            self.logger.record("train/latent_target_std", float(nominal_std))
            self.logger.record("train/action_edge_fraction", float((first.abs() > .98).float().mean()))
        elapsed = perf_counter() - update_started
        self.logger.record("time/update_seconds", elapsed)
        if hasattr(self, "_sampling_seconds"):
            self.logger.record("time/sampling_fraction", self._sampling_seconds / (self._sampling_seconds + elapsed))

    def _record_gradient_balance(self):
        """Compare gradients, not the misleading absolute loss magnitudes."""
        buffer = self.rollout_buffer
        count = buffer.buffer_size * buffer.n_envs
        indices = np.linspace(0, count - 1, min(512, count), dtype=int)
        obs = torch.as_tensor(buffer.observations.reshape(count, -1)[indices], device=self.device)
        actions = torch.as_tensor(buffer.actions.reshape(count, -1)[indices], device=self.device)
        old_log_prob = torch.as_tensor(buffer.log_probs.reshape(-1)[indices], device=self.device)
        advantages = torch.as_tensor(buffer.advantages.reshape(-1)[indices], device=self.device)
        if self.normalize_advantage and len(indices) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        _, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        if entropy is None:
            return
        ratio = torch.exp(log_prob - old_log_prob)
        clip = self.clip_range(self._current_progress_remaining)
        pg = -torch.minimum(advantages * ratio, advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
        entropy_loss = -self.ent_coef * entropy.mean()
        parameters = list(self.policy.mlp_extractor.policy_net.parameters()) + list(self.policy.action_net.parameters()) + [self.policy.log_std]
        pg_grad = torch.autograd.grad(pg, parameters, retain_graph=True, allow_unused=True)
        entropy_grad = torch.autograd.grad(entropy_loss, parameters, allow_unused=True)
        def norm(gradients):
            return float(torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt())
        pg_norm, entropy_norm = norm(pg_grad), norm(entropy_grad)
        self.logger.record("train/policy_gradient_norm", pg_norm)
        self.logger.record("train/entropy_gradient_norm", entropy_norm)
        self.logger.record("train/entropy_gradient_ratio", entropy_norm / max(pg_norm, 1e-12))
