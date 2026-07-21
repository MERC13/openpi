"""Extract the LIBERO BDDL goal-predicate vector from a live env.

LIBERO's public env API only exposes terminal success (`done`). The per-predicate
progress vector — which we need for the RL state (CLAUDE.md §5 Phase D) and for
audit logging — is not exposed, but is computable from the underlying problem
env, which evaluates each goal predicate individually every step:

    goal_state = problem_env.parsed_problem["goal_state"]   # AND-list of atoms
    vector     = [problem_env._eval_predicate(atom) for atom in goal_state]

Each atom is ``[fn_name, obj]`` (unary, e.g. ["Close", region]) or
``[fn_name, objA, objB]`` (binary, e.g. ["In", obj, region]). Predicates read
live sim state, so calling this at any timestep is valid. They are noisy binary
geometric checks and can flicker/regress step-to-step (CLAUDE.md §9), so callers
that shape reward on them should track a running max, not raw deltas.

This module is duck-typed and imports no libero/MuJoCo — it operates on whatever
problem-env object it is handed, so it is unit-testable locally with a mock.
"""

import numpy as np


def find_problem_env(env):
    """Locate the underlying BDDL problem env from a LIBERO OffScreenRenderEnv.

    The problem env is the object carrying ``parsed_problem`` and
    ``_eval_predicate``. LIBERO wraps it (``OffScreenRenderEnv`` -> ``.env``), but
    layouts vary, so walk a small chain of ``.env`` attributes and return the
    first object that looks like a problem env.
    """
    candidate = env
    for _ in range(4):
        if hasattr(candidate, "parsed_problem") and hasattr(candidate, "_eval_predicate"):
            return candidate
        if hasattr(candidate, "env"):
            candidate = candidate.env
        else:
            break
    raise AttributeError("could not locate a BDDL problem env (no parsed_problem/_eval_predicate found)")


def goal_state(problem_env) -> list:
    """The parsed goal as an AND-list of predicate atoms."""
    return problem_env.parsed_problem["goal_state"]


def predicate_labels(problem_env) -> list[str]:
    """Human-readable ``fn(obj[, obj])`` label per goal predicate, in order."""
    return [f"{atom[0]}({', '.join(str(a) for a in atom[1:])})" for atom in goal_state(problem_env)]


def predicate_vector(problem_env) -> np.ndarray:
    """Float32 vector of per-predicate satisfaction (1.0 satisfied, 0.0 not)."""
    atoms = goal_state(problem_env)
    # _eval_predicate is LIBERO's per-atom checker; it is "private" but the only
    # way to read per-predicate progress (the public API exposes only terminal done).
    return np.asarray(
        [float(bool(problem_env._eval_predicate(atom))) for atom in atoms],  # noqa: SLF001
        dtype=np.float32,
    )
