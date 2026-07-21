"""Unit tests for the BDDL predicate-vector wrapper (mock env, no MuJoCo)."""

import numpy as np
import pytest

from steering_rl.libero import predicates


class FakeProblemEnv:
    """Duck-types the LIBERO problem env: parsed_problem + _eval_predicate."""

    def __init__(self, goal_state, satisfied):
        self.parsed_problem = {"goal_state": goal_state}
        self._satisfied = satisfied  # dict: tuple(atom) -> bool

    def _eval_predicate(self, atom):
        return self._satisfied[tuple(atom)]


class Wrapper:
    """Duck-types OffScreenRenderEnv nesting: exposes an inner `.env`."""

    def __init__(self, inner):
        self.env = inner


def _make(goal_state, sat):
    return FakeProblemEnv(goal_state, sat)


def test_predicate_vector():
    goal = [["In", "soup", "basket"], ["In", "sauce", "basket"]]
    prob = _make(goal, {("In", "soup", "basket"): True, ("In", "sauce", "basket"): False})
    vec = predicates.predicate_vector(prob)
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, np.array([1.0, 0.0], dtype=np.float32))


def test_predicate_labels():
    goal = [["In", "bowl", "drawer"], ["Close", "drawer"]]
    prob = _make(goal, {("In", "bowl", "drawer"): False, ("Close", "drawer"): True})
    assert predicates.predicate_labels(prob) == ["In(bowl, drawer)", "Close(drawer)"]


def test_find_problem_env_through_wrapper_chain():
    goal = [["Close", "microwave"]]
    prob = _make(goal, {("Close", "microwave"): True})
    wrapped = Wrapper(Wrapper(prob))  # two layers of .env nesting
    assert predicates.find_problem_env(wrapped) is prob


def test_find_problem_env_returns_self_when_it_is_the_problem():
    prob = _make([["Close", "x"]], {("Close", "x"): False})
    assert predicates.find_problem_env(prob) is prob


def test_find_problem_env_raises_when_absent():
    class Bare:
        pass

    with pytest.raises(AttributeError, match="problem env"):
        predicates.find_problem_env(Bare())
