"""Unit tests for the Phase C report renderer (pure formatting over records)."""

from steering_rl.libero import menus
from steering_rl.libero import report
from steering_rl.libero import tasks

TASK = tasks.TASK_NAMES[0]


def _records(prompt, category, n_success, n_total):
    return [
        {"task": TASK, "prompt": prompt, "category": category, "episode_index": i, "success": i < n_success}
        for i in range(n_total)
    ]


def _sample():
    records = []
    records += _records("canonical", menus.CANONICAL, 3, 4)  # 0.75
    records += _records("subgoal", menus.SUBGOAL, 4, 4)  # 1.00
    records += _records("misleading", menus.MISLEADING, 1, 4)  # 0.25
    return records


def test_format_report_contains_rates_and_verdict():
    text = report.format_report(_sample(), spread_threshold=0.15, margin=0.0)
    assert TASK in text
    assert "1.00 (4/4)" in text  # subgoal
    assert "0.75 (3/4)" in text  # canonical
    assert "0.25 (1/4)" in text  # misleading
    assert "#" * report._BAR_WIDTH in text  # a full bar rendered for the 1.00 prompt  # noqa: SLF001
    # spread across action prompts = 1.00 - 0.75 = 0.25 >= 0.15, correct beats misleading
    assert "gate: PASS" in text
    assert "GATE (all tasks pass): True" in text


def test_flat_report_fails_gate():
    records = []
    for prompt, cat in [("c", menus.CANONICAL), ("s", menus.SUBGOAL), ("m", menus.MISLEADING)]:
        records += _records(prompt, cat, 3, 4)  # all 0.75
    text = report.format_report(records, spread_threshold=0.15, margin=0.0)
    assert "gate: fail" in text
    assert "GATE (all tasks pass): False" in text
    assert "pi05_base / pi05_droid" in text
