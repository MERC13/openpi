"""Unit tests for the fake steering env."""

import numpy as np
import pytest

from steering_rl.fake_env import FakeEnvConfig
from steering_rl.fake_env import FakeSteeringEnv


def test_reset_obs_shape():
    env = FakeSteeringEnv(FakeEnvConfig(num_stages=3, clip_dim=16))
    obs = env.reset()
    assert obs.shape == (16 + 3,)
    assert obs.dtype == np.float32
    # predicate slice starts all-zero at reset
    np.testing.assert_array_equal(obs[16:], np.zeros(3, dtype=np.float32))


def test_step_before_reset_raises():
    env = FakeSteeringEnv()
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("mode", ["sparse", "shaped"])
def test_expert_policy_solves_in_k_steps(mode):
    cfg = FakeEnvConfig(num_stages=3, reward_mode=mode)
    env = FakeSteeringEnv(cfg)
    env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done:
        obs, reward, done, info = env.step(env.expert_action())
        total += reward
        steps += 1
    assert info["success"] is True
    assert steps == cfg.num_stages  # optimal solves in exactly K decisions
    assert total == pytest.approx(env.optimal_return())


def test_wrong_action_makes_no_progress():
    env = FakeSteeringEnv(FakeEnvConfig(num_stages=3, menu_size=5))
    env.reset()
    wrong = (env.expert_action() + 1) % env.num_actions
    obs, reward, done, info = env.step(wrong)
    assert info["newly_satisfied"] == 0
    assert info["stage"] == 0
    assert reward == 0.0
    assert done is False


def test_no_fixed_action_can_solve():
    # With distinct per-stage correct actions, no single fixed action solves the
    # multi-stage task — this is the property that makes state-dependent steering
    # necessary (the core hypothesis in miniature).
    env = FakeSteeringEnv(FakeEnvConfig(num_stages=3, menu_size=5, max_steps=20))
    for fixed_action in range(env.num_actions):
        env.reset()
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(fixed_action)
        assert info["success"] is False


def test_shaped_reward_per_predicate():
    cfg = FakeEnvConfig(num_stages=3, reward_mode="shaped", shaping_coef=0.25)
    env = FakeSteeringEnv(cfg)
    env.reset()
    # first correct step satisfies one predicate but not the whole task
    _, reward, _, info = env.step(env.expert_action())
    assert info["newly_satisfied"] == 1
    assert info["success"] is False
    assert reward == pytest.approx(0.25)


def test_truncation_reports_not_success():
    cfg = FakeEnvConfig(num_stages=3, menu_size=5, max_steps=2)
    env = FakeSteeringEnv(cfg)
    env.reset()
    wrong = (env.expert_action() + 1) % env.num_actions
    done = False
    info = {}
    steps = 0
    while not done:
        _, _, done, info = env.step(wrong)
        steps += 1
    assert steps == 2
    assert info["success"] is False
    assert info["truncated"] is True


def test_determinism_same_seed():
    a = FakeSteeringEnv(FakeEnvConfig(seed=7))
    b = FakeSteeringEnv(FakeEnvConfig(seed=7))
    np.testing.assert_array_equal(a.correct_seq, b.correct_seq)
    np.testing.assert_allclose(a.reset(), b.reset())


def test_different_seeds_differ():
    a = FakeSteeringEnv(FakeEnvConfig(seed=1, num_stages=4, menu_size=6))
    b = FakeSteeringEnv(FakeEnvConfig(seed=2, num_stages=4, menu_size=6))
    assert not np.array_equal(a.correct_seq, b.correct_seq)


def test_invalid_action_raises():
    env = FakeSteeringEnv(FakeEnvConfig(menu_size=5))
    env.reset()
    with pytest.raises(ValueError, match="out of range"):
        env.step(5)


@pytest.mark.parametrize("bad", [{"num_stages": 0}, {"menu_size": 1}, {"reward_mode": "dense"}])
def test_config_validation(bad):
    with pytest.raises(ValueError, match="must be"):
        FakeEnvConfig(**bad)
