"""Unit tests for the PPO machinery, including the Phase-A exit criterion."""

import numpy as np
import pytest
import torch

from steering_rl.fake_env import FakeEnvConfig
from steering_rl.fake_env import FakeSteeringEnv
from steering_rl.policy import SteeringPolicy
from steering_rl.ppo import PPOConfig
from steering_rl.ppo import compute_gae
from steering_rl.ppo import train_ppo


def test_compute_gae_terminal_hand_computed():
    # T=2, terminal episode, gamma=0.9, lambda=0.5.
    # t=1: delta = 1 + 0 - 0.8 = 0.2                -> A1 = 0.2
    # t=0: delta = 0 + 0.9*0.8 - 0.5 = 0.22         -> A0 = 0.22 + 0.9*0.5*0.2 = 0.31
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.8], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, last_value=0.0, terminal=True, gamma=0.9, gae_lambda=0.5)
    np.testing.assert_allclose(adv, [0.31, 0.2], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-6)


def test_compute_gae_truncation_bootstraps():
    # T=1, truncated (not terminal): delta = 1 + 0.9*2.0 - 0.5 = 2.3
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    adv, ret = compute_gae(rewards, values, last_value=2.0, terminal=False, gamma=0.9, gae_lambda=0.5)
    np.testing.assert_allclose(adv, [2.3], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ret, [2.8], rtol=1e-5, atol=1e-6)


def test_compute_gae_returns_equal_adv_plus_values():
    rng = np.random.default_rng(0)
    rewards = rng.standard_normal(10).astype(np.float32)
    values = rng.standard_normal(10).astype(np.float32)
    adv, ret = compute_gae(rewards, values, last_value=0.3, terminal=False, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(ret, adv + values, rtol=1e-6)


def test_policy_shapes():
    torch.manual_seed(0)
    policy = SteeringPolicy(obs_dim=19, num_actions=5, hidden_dim=32)
    obs = torch.randn(8, 19)
    out = policy.act(obs)
    assert out["action"].shape == (8,)
    assert out["log_prob"].shape == (8,)
    assert out["value"].shape == (8,)

    evaluated = policy.evaluate_actions(obs, out["action"])
    assert evaluated["log_prob"].shape == (8,)
    assert evaluated["value"].shape == (8,)
    # value head is differentiable
    evaluated["value"].sum().backward()


def test_ppo_learns_toy_task():
    """Phase-A exit criterion: PPO drives fake-env success to near-optimal on CPU."""
    env = FakeSteeringEnv(FakeEnvConfig(num_stages=3, menu_size=5, reward_mode="sparse"))
    config = PPOConfig(num_updates=30, rollout_episodes=24, seed=0, device="cpu")
    _, history = train_ppo(env, config)

    early = history["success_rate"][0]
    final = float(np.mean(history["success_rate"][-5:]))
    assert final >= 0.95, f"PPO failed to learn: final success {final:.2f} (early {early:.2f})"
    assert final > early  # learning actually happened
    # mean return should approach the optimal return
    assert np.mean(history["mean_return"][-5:]) >= 0.95 * history["optimal_return"]


@pytest.mark.parametrize("mode", ["sparse", "shaped"])
def test_ppo_learns_both_reward_modes(mode):
    env = FakeSteeringEnv(FakeEnvConfig(num_stages=3, menu_size=5, reward_mode=mode))
    config = PPOConfig(num_updates=25, rollout_episodes=24, seed=1, device="cpu")
    _, history = train_ppo(env, config)
    assert float(np.mean(history["success_rate"][-5:])) >= 0.9
