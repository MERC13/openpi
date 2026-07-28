"""The locked Phase-C evaluation set and protocol constants.

Per CLAUDE.md §7, these are fixed ONCE and reused verbatim across every
condition (audit, fixed-prompt baselines, RL eval, final comparison). Changing
this file changes the whole eval; don't edit it casually.

The 5 tasks are all multi-stage LIBERO-10 pick-and-place-into-container tasks
with exactly TWO goal predicates each (verified against their BDDL
``:goal`` blocks), so the predicate vector is uniform length-2 across the set.
Canonical prompts are exactly the strings LIBERO feeds the policy — i.e.
``task.language`` (parsed from the filename by ``grab_language_from_filename``).
"""

from __future__ import annotations  # 3.8 client venv: keep PEP 585/604 generics lazy

import dataclasses

TASK_SUITE_NAME = "libero_10"

# libero_10 renders long; matches examples/libero/main.py for this suite.
MAX_STEPS = 520
NUM_STEPS_WAIT = 10  # sim settle steps before the first real action

# Eval protocol (CLAUDE.md §7). 20 episodes per task, fixed init-state indices,
# fixed env seed, reused across ALL conditions.
NUM_EPISODES = 20
EPISODE_INDICES = tuple(range(NUM_EPISODES))  # index into task_suite.get_task_init_states
ENV_SEED = 7  # matches openpi's example default; affects object placement


@dataclasses.dataclass(frozen=True)
class LiberoTask:
    """Identity of one locked task."""

    name: str  # LIBERO benchmark task name (== bddl filename stem)
    canonical: str  # the prompt LIBERO normally feeds (task.language)
    predicate_labels: tuple[str, ...]  # human-readable goal predicates, in goal_state order


# Order is fixed; index in this tuple is the task's stable id for logging.
TASKS: tuple[LiberoTask, ...] = (
    LiberoTask(
        name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        canonical="put both the alphabet soup and the tomato sauce in the basket",
        predicate_labels=("In(alphabet_soup, basket)", "In(tomato_sauce, basket)"),
    ),
    LiberoTask(
        name="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        canonical="put both the alphabet soup and the cream cheese box in the basket",
        predicate_labels=("In(alphabet_soup, basket)", "In(cream_cheese, basket)"),
    ),
    LiberoTask(
        name="LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        canonical="put both the cream cheese box and the butter in the basket",
        predicate_labels=("In(cream_cheese, basket)", "In(butter, basket)"),
    ),
    LiberoTask(
        name="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        canonical="put the black bowl in the bottom drawer of the cabinet and close it",
        predicate_labels=("Close(cabinet_bottom_drawer)", "In(black_bowl, cabinet_bottom_drawer)"),
    ),
    LiberoTask(
        name="KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        canonical="put the yellow and white mug in the microwave and close it",
        predicate_labels=("In(yellow_white_mug, microwave)", "Close(microwave)"),
    ),
)

TASK_NAMES: tuple[str, ...] = tuple(t.name for t in TASKS)


# Candidate tasks screened AFTER the first audit (2026-07-27) showed 3/5 locked
# tasks are at the legit-prompt ceiling (no spread). These are the remaining
# libero_10 tasks with exactly 2 goal predicates and, structurally, more room for
# the prompt to matter — the two 2-object/2-destination tasks force the policy to
# disambiguate which object goes where. Not part of the §7 locked set until the
# screen confirms spread; screened via `run_audit.py --candidates`.
CANDIDATE_TASKS: tuple[LiberoTask, ...] = (
    LiberoTask(
        name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        canonical="put the white mug on the left plate and put the yellow and white mug on the right plate",
        predicate_labels=("On(porcelain_mug_1, plate_1)", "On(white_yellow_mug_1, plate_2)"),
    ),
    LiberoTask(
        name="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        canonical="put the white mug on the plate and put the chocolate pudding to the right of the plate",
        predicate_labels=("On(porcelain_mug_1, plate_1)", "On(chocolate_pudding_1, plate_right_region)"),
    ),
    LiberoTask(
        name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        canonical="turn on the stove and put the moka pot on it",
        predicate_labels=("Turnon(flat_stove_1)", "On(moka_pot_1, cook_region)"),
    ),
)

CANDIDATE_TASK_NAMES: tuple[str, ...] = tuple(t.name for t in CANDIDATE_TASKS)


def get_task(name: str) -> LiberoTask:
    for task in (*TASKS, *CANDIDATE_TASKS):
        if task.name == name:
            return task
    raise KeyError(f"{name!r} is not one of the Phase-C tasks or candidates")
