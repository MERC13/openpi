"""Render the Phase C audit output into a readable gate report.

Reads ``audit_records.jsonl`` (the resumable per-episode log written by
``run_audit.py`` and pulled off the Modal volume with ``modal volume get``) and
prints, per locked task:

- per-prompt success rates grouped by category (canonical / paraphrase / subgoal
  / misleading), with an ASCII bar,
- the two gate signals (meaningful action-prompt spread; correct > misleading),
- the overall gate verdict (CLAUDE.md §5 Phase C).

Pure reporting over the measured records — it never invents numbers (CLAUDE.md
§2); an empty/partial log just yields a partial report. Reuses
``run_audit.aggregate_audit`` so the report and the gate agree by construction.

Usage:
    python -m steering_rl.libero.report data/steering_rl/audit/audit_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

from steering_rl.libero import menus
from steering_rl.libero.run_audit import DEFAULT_MARGIN
from steering_rl.libero.run_audit import DEFAULT_SPREAD_THRESHOLD
from steering_rl.libero.run_audit import aggregate_audit

_BAR_WIDTH = 20
_CATEGORY_ORDER = (menus.CANONICAL, menus.PARAPHRASE, menus.SUBGOAL, menus.MISLEADING)


def _bar(rate: float) -> str:
    filled = round(rate * _BAR_WIDTH)
    return "#" * filled + "." * (_BAR_WIDTH - filled)


def load_records(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def format_report(records: list[dict], *, spread_threshold: float, margin: float) -> str:
    summary = aggregate_audit(records, spread_threshold=spread_threshold, margin=margin)
    per_prompt = summary["per_prompt"]
    lines: list[str] = []
    n = len(records)
    lines.append(f"Phase C steerability audit — {n} episodes over {len(summary['per_task'])} task(s)")
    lines.append(f"gate thresholds: spread >= {spread_threshold:.2f}, correct > misleading + {margin:.2f}")

    for task_name, t in summary["per_task"].items():
        lines.append("")
        lines.append(task_name)
        prompts = [a for a in per_prompt if a["task"] == task_name]
        by_cat = {c: [a for a in prompts if a["category"] == c] for c in _CATEGORY_ORDER}
        for cat in _CATEGORY_ORDER:
            lines.extend(
                f"  {cat:10s} {_bar(a['success_rate'])} {a['success_rate']:.2f}"
                f" ({a['successes']}/{a['n']})  {a['prompt']}"
                for a in sorted(by_cat[cat], key=lambda a: -a["success_rate"])
            )
        verdict = "PASS" if t["gate_pass"] else "fail"
        lines.append(
            f"  -> spread={t['action_spread']:.2f} (meaningful={t['meaningful_spread']});"
            f" best_action={t['best_action_rate']:.2f} vs misleading={t['misleading_rate']:.2f}"
            f" (correct>misleading={t['correct_beats_misleading']})  [gate: {verdict}]"
        )

    lines.append("")
    lines.append(f"GATE (all tasks pass): {summary['gate_pass_all']}")
    if not summary["gate_pass_all"]:
        lines.append("Flat/failed on pi05_libero? See CLAUDE.md §9: audit pi05_base / pi05_droid before RL.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Phase C audit gate report.")
    parser.add_argument(
        "records",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("data/steering_rl/audit/audit_records.jsonl"),
        help="path to audit_records.jsonl",
    )
    parser.add_argument("--spread-threshold", type=float, default=DEFAULT_SPREAD_THRESHOLD)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    records = load_records(args.records)
    print(format_report(records, spread_threshold=args.spread_threshold, margin=args.margin))


if __name__ == "__main__":
    main()
