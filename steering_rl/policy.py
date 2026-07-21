"""The steering policy: a small actor-critic over [CLIP ‖ predicate] state.

This is the only thing that trains in the project (the pi0.5 VLA is frozen). It
is a 2-layer MLP trunk feeding a categorical policy head over the N-item
instruction menu and a scalar value head for PPO. Deliberately tiny so it runs
on CPU; the real run (Phase D) scales the same architecture to ~1M params over a
real CLIP(ViT-B) embedding.
"""

import torch
from torch.distributions import Categorical
import torch.nn as nn


class SteeringPolicy(nn.Module):
    """Actor-critic MLP: state -> (action logits over the menu, state value)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B, num_actions], value [B])."""
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Sample (or argmax) an action for rollout collection.

        Returns a dict with ``action``, ``log_prob``, ``value``, ``entropy``.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return {
            "action": action,
            "log_prob": dist.log_prob(action),
            "value": value,
            "entropy": dist.entropy(),
        }

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Re-evaluate stored actions under the current policy (PPO update step).

        Returns a dict with ``log_prob``, ``entropy``, ``value`` (all differentiable).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return {
            "log_prob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": value,
        }
