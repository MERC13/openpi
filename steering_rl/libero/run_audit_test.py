"""Unit tests for the audit aggregation (pure function, synthetic records)."""

from steering_rl.libero import menus
from steering_rl.libero import tasks
from steering_rl.libero.run_audit import aggregate_audit

TASK = tasks.TASK_NAMES[0]


def _records(prompt, category, n_success, n_total):
    return [
        {"task": TASK, "prompt": prompt, "category": category, "episode_index": i, "success": i < n_success}
        for i in range(n_total)
    ]


def _steerable_records():
    records = []
    records += _records("canonical", menus.CANONICAL, 3, 4)  # 0.75
    records += _records("para", menus.PARAPHRASE, 2, 4)  # 0.50
    records += _records("subgoal", menus.SUBGOAL, 4, 4)  # 1.00
    records += _records("misleading", menus.MISLEADING, 1, 4)  # 0.25
    return records


def test_per_prompt_success_rates():
    summary = aggregate_audit(_steerable_records())
    rates = {a["prompt"]: a["success_rate"] for a in summary["per_prompt"]}
    assert rates == {"canonical": 0.75, "para": 0.5, "subgoal": 1.0, "misleading": 0.25}


def test_per_task_metrics_and_gate_pass():
    summary = aggregate_audit(_steerable_records(), spread_threshold=0.15, margin=0.0)
    t = summary["per_task"][TASK]
    assert t["canonical_rate"] == 0.75
    assert t["best_action_rate"] == 1.0
    assert t["best_action_prompt"] == "subgoal"
    assert t["misleading_rate"] == 0.25
    assert t["action_spread"] == 0.5  # 1.0 - 0.5
    assert t["meaningful_spread"] is True
    assert t["correct_beats_misleading"] is True
    assert t["gate_pass"] is True
    assert summary["gate_pass_all"] is True


def test_errored_records_are_ignored():
    # A clean 1.0 measurement plus 3 connection-errored records for the same prompt
    # must aggregate to rate 1.0 over n=1 — errored episodes are not counted.
    records = [{"task": TASK, "prompt": "p", "category": menus.CANONICAL, "episode_index": 0, "success": True}]
    records += [
        {"task": TASK, "prompt": "p", "category": menus.CANONICAL, "episode_index": i, "success": False, "error": "conn"}
        for i in range(1, 4)
    ]
    summary = aggregate_audit(records)
    agg = summary["per_prompt"][0]
    assert agg["n"] == 1
    assert agg["success_rate"] == 1.0


def test_flat_interface_fails_gate():
    # All prompts identical success -> no spread, correct does not beat misleading.
    records = []
    for prompt, cat in [("c", menus.CANONICAL), ("p", menus.PARAPHRASE), ("s", menus.SUBGOAL), ("m", menus.MISLEADING)]:
        records += _records(prompt, cat, 3, 4)  # every prompt 0.75
    summary = aggregate_audit(records, spread_threshold=0.15, margin=0.0)
    t = summary["per_task"][TASK]
    assert t["action_spread"] == 0.0
    assert t["meaningful_spread"] is False
    assert t["correct_beats_misleading"] is False  # 0.75 > 0.75 is False
    assert t["gate_pass"] is False
    assert summary["gate_pass_all"] is False
