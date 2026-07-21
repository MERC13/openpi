# steering_rl

RL over the **prompt interface** of a frozen π0.5 VLA. See the repo-root
`CLAUDE.md` for the full project brief. This package holds only our code; openpi
is treated as a frozen dependency and is never edited.

## Phase A (current): local PPO machinery vs. a fake env

Everything here runs locally on CPU with **no π0.5, MuJoCo, or GPU dependency** —
it exists to validate the PPO loop, policy net, and reward logic before any of it
touches the real (cloud-served) VLA.

| Module | What it is |
| --- | --- |
| `fake_env.py` | `FakeSteeringEnv`: a mock LIBERO-like semi-MDP. State = `[stub-CLIP embedding ‖ predicate vector]`; action = index into an N-item instruction menu; multi-stage task where the correct instruction differs per stage, so **no fixed action can solve it — only state-dependent steering can**. Sparse or shaped reward. |
| `policy.py` | `SteeringPolicy`: a 2-layer MLP actor-critic over the state → categorical over the menu + value head. |
| `ppo.py` | PPO + GAE (`compute_gae` is a pure, unit-tested function; `train_ppo` drives the loop). Nothing here assumes the fake env — Phase D swaps in the real LIBERO client → frozen π0.5 server behind the same API. |

**Exit criterion (met):** PPO drives fake-env success to near-optimal on CPU.

### Run

```bash
# quick demo (prints the learning curve)
uv run python -m steering_rl.ppo

# tests (fake_env + ppo, incl. the exit-criterion integration test)
uv run python -m pytest steering_rl/
```

Note: `steering_rl/` is not in openpi's `pyproject.toml` `testpaths`, so run its
tests with the explicit path above rather than a bare `uv run pytest`.
