"""PPO + GAE for the steering policy (Phase A machinery).

A readable, CS285-style clipped-PPO implementation with generalized advantage
estimation. Kept deliberately simple and unit-testable: :func:`compute_gae` is a
pure function, and :func:`train_ppo` drives the whole loop against any env with
the :class:`~steering_rl.fake_env.FakeSteeringEnv` API. In Phase D the same loop
collects rollouts through the frozen pi0.5 websocket server instead of the fake
env; nothing here assumes the fake env specifically.
"""

import dataclasses
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from steering_rl.policy import SteeringPolicy


@dataclasses.dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-3
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_episodes: int = 32  # full episodes collected per update
    num_updates: int = 60
    hidden_dim: int = 64
    seed: int = 0
    device: str = "cpu"  # tiny nets — CPU is fast and deterministic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
    *,
    terminal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation for a SINGLE episode.

    Args:
        rewards: shape ``[T]`` rewards ``r_0..r_{T-1}``.
        values: shape ``[T]`` value estimates ``V(s_0)..V(s_{T-1})``.
        last_value: bootstrap value ``V(s_T)`` of the state after the last step.
            Ignored (treated as 0) when ``terminal`` is True.
        terminal: True iff the episode ended in a true environment terminal
            (e.g. task success). False for a time-limit truncation, where the
            future value is bootstrapped from ``last_value``.
        gamma: discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)``, each shape ``[T]`` and float32.
        ``returns == advantages + values`` (the value-function targets).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    length = len(rewards)
    advantages = np.zeros(length, dtype=np.float32)

    last_gae = 0.0
    for t in reversed(range(length)):
        if t == length - 1:
            next_nonterminal = 0.0 if terminal else 1.0
            next_value = 0.0 if terminal else float(last_value)
        else:
            next_nonterminal = 1.0
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def collect_rollouts(env, policy: SteeringPolicy, config: PPOConfig) -> tuple[dict, dict]:
    """Run ``config.rollout_episodes`` full episodes; return a flat batch + stats."""
    device = torch.device(config.device)
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    logp_buf: list[float] = []
    adv_buf: list[np.ndarray] = []
    ret_buf: list[np.ndarray] = []

    ep_returns: list[float] = []
    ep_successes: list[float] = []

    for _ in range(config.rollout_episodes):
        obs = env.reset()
        ep_obs, ep_act, ep_logp, ep_val, ep_rew = [], [], [], [], []
        terminal = False
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            out = policy.act(obs_t)
            action = int(out["action"].item())
            next_obs, reward, done, info = env.step(action)

            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(action)
            ep_logp.append(float(out["log_prob"].item()))
            ep_val.append(float(out["value"].item()))
            ep_rew.append(float(reward))

            obs = next_obs
            if done:
                terminal = bool(info.get("success", False))
                break

        if terminal:
            last_value = 0.0
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                last_value = float(policy.forward(obs_t)[1].item())

        advantages, returns = compute_gae(
            np.asarray(ep_rew, dtype=np.float32),
            np.asarray(ep_val, dtype=np.float32),
            last_value,
            config.gamma,
            config.gae_lambda,
            terminal=terminal,
        )

        obs_buf.extend(ep_obs)
        act_buf.extend(ep_act)
        logp_buf.extend(ep_logp)
        adv_buf.append(advantages)
        ret_buf.append(returns)

        ep_returns.append(float(sum(ep_rew)))
        ep_successes.append(float(terminal))

    batch = {
        "obs": torch.as_tensor(np.stack(obs_buf), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(np.concatenate(adv_buf), dtype=torch.float32, device=device),
        "returns": torch.as_tensor(np.concatenate(ret_buf), dtype=torch.float32, device=device),
    }
    stats = {
        "mean_return": float(np.mean(ep_returns)),
        "success_rate": float(np.mean(ep_successes)),
    }
    return batch, stats


def ppo_update(policy: SteeringPolicy, optimizer: torch.optim.Optimizer, batch: dict, config: PPOConfig) -> dict:
    """One PPO update: several epochs of minibatch SGD on the clipped objective."""
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    returns = batch["returns"]

    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    num_samples = obs.shape[0]
    last_metrics: dict = {}
    for _ in range(config.update_epochs):
        perm = torch.randperm(num_samples, device=obs.device)
        for start in range(0, num_samples, config.minibatch_size):
            mb = perm[start : start + config.minibatch_size]
            evaluated = policy.evaluate_actions(obs[mb], actions[mb])

            ratio = torch.exp(evaluated["log_prob"] - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (evaluated["value"] - returns[mb]).pow(2).mean()
            entropy = evaluated["entropy"].mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            last_metrics = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }
    return last_metrics


def train_ppo(env, config: PPOConfig | None = None) -> tuple[SteeringPolicy, dict]:
    """Train a :class:`SteeringPolicy` on ``env`` with PPO. Returns (policy, history)."""
    config = config or PPOConfig()
    set_seed(config.seed)

    device = torch.device(config.device)
    policy = SteeringPolicy(env.obs_dim, env.num_actions, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    history: dict = {
        "update": [],
        "mean_return": [],
        "success_rate": [],
        "optimal_return": env.optimal_return(),
    }
    for update in range(config.num_updates):
        batch, stats = collect_rollouts(env, policy, config)
        ppo_update(policy, optimizer, batch, config)
        history["update"].append(update)
        history["mean_return"].append(stats["mean_return"])
        history["success_rate"].append(stats["success_rate"])

    return policy, history


def _demo() -> None:
    from steering_rl.fake_env import FakeEnvConfig
    from steering_rl.fake_env import FakeSteeringEnv

    env = FakeSteeringEnv(FakeEnvConfig(reward_mode="sparse"))
    _, history = train_ppo(env, PPOConfig())
    opt = history["optimal_return"]
    for u in range(0, len(history["update"]), 10):
        print(
            f"update {u:3d} | mean_return {history['mean_return'][u]:.3f} "
            f"/ optimal {opt:.3f} | success {history['success_rate'][u]:.2f}"
        )
    print(f"final success rate: {history['success_rate'][-1]:.2f}")


if __name__ == "__main__":
    _demo()
