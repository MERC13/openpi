"""Unit tests for the per-task instruction menus."""

import pytest

from steering_rl.libero import menus
from steering_rl.libero import tasks


def test_every_locked_and_candidate_task_has_a_menu():
    assert set(tasks.TASK_NAMES) <= set(menus.MENUS)
    assert set(tasks.CANDIDATE_TASK_NAMES) <= set(menus.MENUS)


@pytest.mark.parametrize("task_name", tasks.CANDIDATE_TASK_NAMES)
def test_candidate_menu_structure(task_name):
    menu = menus.menu(task_name)
    cats = [item.category for item in menu]
    assert all(c in menus.CATEGORIES for c in cats)
    assert cats.count(menus.CANONICAL) == 1
    assert cats.count(menus.SUBGOAL) >= 2
    assert cats.count(menus.MISLEADING) >= 1
    texts = [item.text for item in menu]
    assert len(texts) == len(set(texts)), "menu has duplicate prompt strings"
    assert menus.canonical_prompt(task_name) == tasks.get_task(task_name).canonical


@pytest.mark.parametrize("task_name", tasks.TASK_NAMES)
def test_menu_structure(task_name):
    menu = menus.menu(task_name)
    cats = [item.category for item in menu]

    assert all(c in menus.CATEGORIES for c in cats)
    assert cats.count(menus.CANONICAL) == 1
    assert cats.count(menus.PARAPHRASE) >= 3
    assert cats.count(menus.SUBGOAL) >= 2
    assert cats.count(menus.MISLEADING) >= 1

    texts = [item.text for item in menu]
    assert len(texts) == len(set(texts)), "menu has duplicate prompt strings"


@pytest.mark.parametrize("task_name", tasks.TASK_NAMES)
def test_canonical_matches_locked_task_language(task_name):
    # The canonical menu entry must equal exactly what LIBERO feeds (task.language).
    assert menus.canonical_prompt(task_name) == tasks.get_task(task_name).canonical


@pytest.mark.parametrize("task_name", tasks.TASK_NAMES)
def test_action_menu_excludes_misleading(task_name):
    action = menus.action_menu(task_name)
    assert all(item.category != menus.MISLEADING for item in action)
    full = menus.menu(task_name)
    n_misleading = sum(item.category == menus.MISLEADING for item in full)
    assert len(action) == len(full) - n_misleading
