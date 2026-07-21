"""A mock LIBERO-like environment for validating the steering-RL machinery.

This env exists ONLY so the PPO loop, the policy net, and the reward logic can
be developed and unit-tested locally with zero pi0.5 / MuJoCo / GPU dependency
(Phase A of the plan in CLAUDE.md). It mimics the *structure* of the real setup,
not its content:

- **State** = [stub-CLIP image embedding ‖ BDDL-style predicate vector]. The
  image embedding stands in for a real CLIP(ViT-B) frame embedding; here it is a
  fixed random projection of the current stage plus Gaussian noise. The predicate
  vector stands in for the LIBERO goal predicates (`parsed_problem["goal_state"]`).
- **Action** = an index into an N-item instruction menu. This is the toy analogue
  of choosing which prompt string to hand the frozen VLA.
- **Dynamics** = a multi-stage task. At each stage exactly one menu item is the
  "correct instruction" that satisfies the next predicate, and the correct item is
  DIFFERENT across stages. Therefore no single fixed action can solve the task —
  only a state-dependent policy can. This is the toy analogue of the core
  hypothesis: learned steering can beat any best-fixed prompt.
- **Reward** = terminal success (all predicates satisfied), with an optional dense
  shaping term per newly-satisfied predicate.

One `env.step()` == one decision point (one replan boundary in the real system).

The env is numpy-only and deterministic given its seed.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class FakeEnvConfig:
    """Configuration for :class:`FakeSteeringEnv`."""

    num_stages: int = 3  # number of predicates / task stages (K)
    menu_size: int = 5  # size of the instruction menu (N == number of actions)
    clip_dim: int = 16  # dimensionality of the stub-CLIP image embedding
    max_steps: int = 12  # episode truncation horizon (decision points)
    reward_mode: str = "sparse"  # "sparse" (success only) or "shaped" (+ per-predicate)
    shaping_coef: float = 0.25  # reward per newly-satisfied predicate in "shaped" mode
    success_bonus: float = 1.0  # terminal reward when all predicates are satisfied
    wrong_action_penalty: float = 0.0  # optional penalty for a non-advancing action
    image_noise_std: float = 0.1  # Gaussian noise added to the stub-CLIP embedding
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.menu_size < 2:
            raise ValueError("menu_size must be >= 2 (need a real choice)")
        if self.reward_mode not in ("sparse", "shaped"):
            raise ValueError(f"reward_mode must be 'sparse' or 'shaped', got {self.reward_mode!r}")


class FakeSteeringEnv:
    """A deterministic, numpy-only multi-stage steering task.

    Gym-like API (no gym dependency): ``reset() -> obs`` and
    ``step(action) -> (obs, reward, done, info)``. The observation is a float32
    vector of length ``clip_dim + num_stages``.
    """

    def __init__(self, config: FakeEnvConfig | None = None):
        self.config = config or FakeEnvConfig()
        self._rng = np.random.default_rng(self.config.seed)

        k = self.config.num_stages
        n = self.config.menu_size

        # The hidden "correct instruction" for each stage. Sampled WITHOUT
        # replacement when the menu is large enough, so every stage needs a
        # different action and no fixed action can satisfy more than one stage.
        if k <= n:
            self.correct_seq = self._rng.permutation(n)[:k].astype(np.int64)
        else:
            # More stages than menu items: repeats are unavoidable, but we still
            # forbid two ADJACENT stages sharing an action to keep it non-trivial.
            seq = [int(self._rng.integers(n))]
            while len(seq) < k:
                nxt = int(self._rng.integers(n))
                if nxt != seq[-1]:
                    seq.append(nxt)
            self.correct_seq = np.asarray(seq, dtype=np.int64)

        # Fixed random projection giving each stage (0..K) a distinct, stable
        # "visual" embedding — the stub for a CLIP frame embedding.
        self._clip_table = self._rng.standard_normal((k + 1, self.config.clip_dim)).astype(np.float32)

        self._predicates = np.zeros(k, dtype=np.float32)
        self._t = 0
        self._done = True  # force a reset() before stepping

    # -- properties -------------------------------------------------------

    @property
    def num_actions(self) -> int:
        return self.config.menu_size

    @property
    def num_stages(self) -> int:
        return self.config.num_stages

    @property
    def clip_dim(self) -> int:
        return self.config.clip_dim

    @property
    def obs_dim(self) -> int:
        return self.config.clip_dim + self.config.num_stages

    @property
    def current_stage(self) -> int:
        """Number of predicates satisfied so far (they are satisfied in order)."""
        return int(self._predicates.sum())

    # -- core API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._predicates = np.zeros(self.config.num_stages, dtype=np.float32)
        self._t = 0
        self._done = False
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action {action} out of range [0, {self.num_actions})")

        stage = self.current_stage
        newly_satisfied = 0
        if stage < self.config.num_stages and action == int(self.correct_seq[stage]):
            self._predicates[stage] = 1.0
            newly_satisfied = 1

        self._t += 1
        success = self.current_stage == self.config.num_stages
        truncated = self._t >= self.config.max_steps
        self._done = bool(success or truncated)

        reward = 0.0
        if self.config.reward_mode == "shaped":
            reward += self.config.shaping_coef * newly_satisfied
        if newly_satisfied == 0 and self.config.wrong_action_penalty:
            reward -= self.config.wrong_action_penalty
        if success:
            reward += self.config.success_bonus

        info = {
            "success": success,
            "truncated": truncated and not success,
            "stage": self.current_stage,
            "newly_satisfied": newly_satisfied,
        }
        return self._observation(), float(reward), self._done, info

    # -- helpers ----------------------------------------------------------

    def expert_action(self) -> int:
        """The optimal action for the current state (for tests / sanity checks)."""
        stage = self.current_stage
        if stage >= self.config.num_stages:
            return 0  # task already solved; any action is fine
        return int(self.correct_seq[stage])

    def optimal_return(self) -> float:
        """Best achievable undiscounted episode return (an optimal policy hits this)."""
        total = self.config.success_bonus
        if self.config.reward_mode == "shaped":
            total += self.config.shaping_coef * self.config.num_stages
        return float(total)

    def _observation(self) -> np.ndarray:
        emb = self._clip_table[self.current_stage].copy()
        if self.config.image_noise_std > 0:
            emb = emb + self.config.image_noise_std * self._rng.standard_normal(self.clip_dim).astype(np.float32)
        return np.concatenate([emb.astype(np.float32), self._predicates.copy()])
