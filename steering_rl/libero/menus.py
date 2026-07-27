"""Per-task instruction menus for the steerability audit (Phase C) and RL (D/E).

Each locked task gets a menu of candidate prompt strings in four categories:

- ``canonical``: exactly the string LIBERO normally feeds (== task.language).
- ``paraphrase``: same-meaning rewrites of the canonical instruction.
- ``subgoal``: decomposed single-stage instructions (one object / one phase).
  These are the interesting steering lever on multi-stage tasks — telling the
  frozen policy which sub-task to focus on *now*.
- ``misleading``: an instruction for a DIFFERENT goal (usually another task's
  string, or the wrong objects). Used only to test the audit requirement
  "correct-instruction > misleading-instruction" (CLAUDE.md §5 Phase C). These
  are excluded from the RL action space by :func:`action_menu`.

The reward always comes from the env's own goal check, never from the prompt
(CLAUDE.md §2) — so a misleading prompt that "succeeds" would be the policy
ignoring the prompt, which is exactly what the audit measures.
"""

from __future__ import annotations  # 3.8 client venv: keep PEP 585/604 generics lazy

import dataclasses

from steering_rl.libero import tasks

CANONICAL = "canonical"
PARAPHRASE = "paraphrase"
SUBGOAL = "subgoal"
MISLEADING = "misleading"

CATEGORIES = (CANONICAL, PARAPHRASE, SUBGOAL, MISLEADING)
# Categories that form the RL policy's action space (misleading probes excluded).
ACTION_CATEGORIES = (CANONICAL, PARAPHRASE, SUBGOAL)


@dataclasses.dataclass(frozen=True)
class PromptItem:
    text: str
    category: str


def _menu(canonical: str, paraphrases: list[str], subgoals: list[str], misleading: list[str]) -> list[PromptItem]:
    items = [PromptItem(canonical, CANONICAL)]
    items += [PromptItem(t, PARAPHRASE) for t in paraphrases]
    items += [PromptItem(t, SUBGOAL) for t in subgoals]
    items += [PromptItem(t, MISLEADING) for t in misleading]
    return items


_T1 = tasks.TASKS[0].name
_T2 = tasks.TASKS[1].name
_T3 = tasks.TASKS[2].name
_T4 = tasks.TASKS[3].name
_T5 = tasks.TASKS[4].name

MENUS: dict[str, list[PromptItem]] = {
    _T1: _menu(
        canonical="put both the alphabet soup and the tomato sauce in the basket",
        paraphrases=[
            "place both the alphabet soup and the tomato sauce into the basket",
            "put the alphabet soup and the tomato sauce in the basket",
            "move the alphabet soup and the tomato sauce into the basket",
            "pick up the alphabet soup and the tomato sauce and put them in the basket",
        ],
        subgoals=[
            "put the alphabet soup in the basket",
            "put the tomato sauce in the basket",
            "pick up the tomato sauce and place it in the basket",
        ],
        misleading=[
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "put both the cream cheese box and the butter in the basket",
        ],
    ),
    _T2: _menu(
        canonical="put both the alphabet soup and the cream cheese box in the basket",
        paraphrases=[
            "place both the alphabet soup and the cream cheese box into the basket",
            "put the alphabet soup and the cream cheese box in the basket",
            "move the alphabet soup and the cream cheese box into the basket",
            "pick up the alphabet soup and the cream cheese box and put them in the basket",
        ],
        subgoals=[
            "put the alphabet soup in the basket",
            "put the cream cheese box in the basket",
            "pick up the cream cheese box and place it in the basket",
        ],
        misleading=[
            "put the yellow and white mug in the microwave and close it",
            "put both the cream cheese box and the butter in the basket",
        ],
    ),
    _T3: _menu(
        canonical="put both the cream cheese box and the butter in the basket",
        paraphrases=[
            "place both the cream cheese box and the butter into the basket",
            "put the cream cheese box and the butter in the basket",
            "move the cream cheese box and the butter into the basket",
            "pick up the cream cheese box and the butter and put them in the basket",
        ],
        subgoals=[
            "put the cream cheese box in the basket",
            "put the butter in the basket",
            "pick up the butter and place it in the basket",
        ],
        misleading=[
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "put both the alphabet soup and the tomato sauce in the basket",
        ],
    ),
    _T4: _menu(
        canonical="put the black bowl in the bottom drawer of the cabinet and close it",
        paraphrases=[
            "place the black bowl into the bottom drawer of the cabinet and close the drawer",
            "put the black bowl into the bottom drawer and close it",
            "move the black bowl into the cabinet's bottom drawer and shut it",
            "stow the black bowl in the bottom drawer of the cabinet and close the drawer",
        ],
        subgoals=[
            "put the black bowl in the bottom drawer of the cabinet",
            "close the bottom drawer of the cabinet",
            "pick up the black bowl and place it in the bottom drawer",
        ],
        misleading=[
            "put both the alphabet soup and the tomato sauce in the basket",
            "put the yellow and white mug in the microwave and close it",
        ],
    ),
    _T5: _menu(
        canonical="put the yellow and white mug in the microwave and close it",
        paraphrases=[
            "place the yellow and white mug into the microwave and close the microwave",
            "put the yellow and white mug into the microwave and shut the door",
            "move the yellow and white mug into the microwave and close it",
            "stow the yellow and white mug in the microwave and close the door",
        ],
        subgoals=[
            "put the yellow and white mug in the microwave",
            "close the microwave",
            "pick up the yellow and white mug and place it in the microwave",
        ],
        misleading=[
            "put the black bowl in the bottom drawer of the cabinet and close it",
            "put both the cream cheese box and the butter in the basket",
        ],
    ),
}


def menu(task_name: str) -> list[PromptItem]:
    """Full audit menu (all categories) for a task."""
    return MENUS[task_name]


def action_menu(task_name: str) -> list[PromptItem]:
    """RL action space: the menu minus misleading probes."""
    return [item for item in MENUS[task_name] if item.category in ACTION_CATEGORIES]


def canonical_prompt(task_name: str) -> str:
    return next(item.text for item in MENUS[task_name] if item.category == CANONICAL)
