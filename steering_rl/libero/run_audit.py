"""Phase C steerability audit: sweep the prompt menu, measure success spread.

For each locked task, hold the env + seeds fixed and run every menu prompt across
the fixed episode set, recording the env's own success. Then aggregate into
per-(task, prompt) success rates and the two gate signals (CLAUDE.md §5 Phase C):

  (a) meaningful spread across legit prompt variations, and
  (b) correct-instruction success > misleading-instruction success.

This is measured on the cloud-served frozen pi0.5 — the numbers are never
invented (CLAUDE.md §2). Runs are resumable: each episode is appended to a JSONL
as it completes, and re-running skips (task, prompt, episode) triples already on
disk, so a free-tier timeout loses at most one episode.

Usage (inside the Docker LIBERO runtime, server already up):
    python steering_rl/libero/run_audit.py --host 0.0.0.0 --port 8000
"""

import dataclasses
import json
import logging
import pathlib

from steering_rl.libero import menus
from steering_rl.libero import tasks
from steering_rl.libero.episode import run_episode

logger = logging.getLogger(__name__)

DEFAULT_SPREAD_THRESHOLD = 0.15  # min action-prompt spread to call the interface "usable"
DEFAULT_MARGIN = 0.0  # required (best action rate) - (misleading rate) margin


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5
    resize_size: int = 224
    out_dir: str = "data/steering_rl/audit"
    episodes: int = tasks.NUM_EPISODES
    resume: bool = True
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD
    margin: float = DEFAULT_MARGIN


def _record_key(record: dict) -> tuple:
    return (record["task"], record["prompt"], record["episode_index"])


def _load_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def aggregate_audit(
    records: list[dict],
    *,
    spread_threshold: float = DEFAULT_SPREAD_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """Reduce raw per-episode records to per-prompt / per-task / gate summaries.

    Pure function: no I/O, no env. ``records`` are dicts with at least
    ``task``, ``prompt``, ``category``, ``success`` (bool).
    """
    # group successes per (task, prompt)
    per_prompt: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["task"], r["prompt"])
        agg = per_prompt.setdefault(
            key, {"task": r["task"], "prompt": r["prompt"], "category": r["category"], "n": 0, "successes": 0}
        )
        agg["n"] += 1
        agg["successes"] += int(bool(r["success"]))
    for agg in per_prompt.values():
        agg["success_rate"] = agg["successes"] / agg["n"] if agg["n"] else 0.0

    per_task: dict[str, dict] = {}
    for task_name in tasks.TASK_NAMES:
        prompts = [a for a in per_prompt.values() if a["task"] == task_name]
        if not prompts:
            continue
        action = [a for a in prompts if a["category"] in menus.ACTION_CATEGORIES]
        misleading = [a for a in prompts if a["category"] == menus.MISLEADING]
        canonical = [a for a in prompts if a["category"] == menus.CANONICAL]

        action_rates = [a["success_rate"] for a in action] or [0.0]
        best_action = max(action, key=lambda a: a["success_rate"]) if action else None
        misleading_rate = sum(a["success_rate"] for a in misleading) / len(misleading) if misleading else float("nan")
        action_spread = max(action_rates) - min(action_rates)
        best_action_rate = max(action_rates)
        correct_beats_misleading = (not misleading) or (best_action_rate > misleading_rate + margin)
        meaningful_spread = action_spread >= spread_threshold

        per_task[task_name] = {
            "canonical_rate": canonical[0]["success_rate"] if canonical else float("nan"),
            "best_action_rate": best_action_rate,
            "best_action_prompt": best_action["prompt"] if best_action else None,
            "misleading_rate": misleading_rate,
            "action_spread": action_spread,
            "meaningful_spread": meaningful_spread,
            "correct_beats_misleading": correct_beats_misleading,
            "gate_pass": meaningful_spread and correct_beats_misleading,
        }

    gate_pass_all = bool(per_task) and all(t["gate_pass"] for t in per_task.values())
    return {
        "per_prompt": sorted(per_prompt.values(), key=lambda a: (a["task"], -a["success_rate"])),
        "per_task": per_task,
        "gate_pass_all": gate_pass_all,
        "spread_threshold": spread_threshold,
        "margin": margin,
    }


def run_audit(args: Args) -> None:
    from openpi_client import websocket_client_policy

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "audit_records.jsonl"

    existing = _load_records(records_path) if args.resume else []
    done_keys = {_record_key(r) for r in existing}
    logger.info("resuming with %d episodes already recorded", len(done_keys))

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    episode_indices = list(tasks.EPISODE_INDICES[: args.episodes])

    with records_path.open("a") as sink:
        for task in tasks.TASKS:
            from steering_rl.libero.episode import make_env

            env, _, initial_states = make_env(task.name, seed=tasks.ENV_SEED)
            for item in menus.menu(task.name):
                for ep_idx in episode_indices:
                    if (task.name, item.text, ep_idx) in done_keys:
                        continue
                    result = run_episode(
                        client,
                        env,
                        initial_states[ep_idx],
                        item.text,
                        max_steps=tasks.MAX_STEPS,
                        replan_steps=args.replan_steps,
                        resize_size=args.resize_size,
                    )
                    record = {
                        "task": task.name,
                        "prompt": item.text,
                        "category": item.category,
                        "episode_index": ep_idx,
                        "success": result.success,
                        "steps": result.steps,
                        "num_replans": result.num_replans,
                        "final_predicates": None
                        if result.final_predicates is None
                        else result.final_predicates.tolist(),
                        "error": result.error,
                    }
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    logger.info("%s | %-60s | ep %2d | success=%s", task.name[:20], item.text, ep_idx, result.success)

    summary = aggregate_audit(_load_records(records_path), spread_threshold=args.spread_threshold, margin=args.margin)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("\n=== Phase C steerability audit ===")
    for task_name, t in summary["per_task"].items():
        print(f"\n{task_name}")
        print(
            f"  canonical={t['canonical_rate']:.2f}  best_action={t['best_action_rate']:.2f}"
            f"  misleading={t['misleading_rate']:.2f}  spread={t['action_spread']:.2f}"
        )
        print(f"  best prompt: {t['best_action_prompt']!r}")
        print(
            f"  meaningful_spread={t['meaningful_spread']}  correct>misleading={t['correct_beats_misleading']}"
            f"  -> gate_pass={t['gate_pass']}"
        )
    print(f"\nGATE (all 5 tasks pass): {summary['gate_pass_all']}")
    if not summary["gate_pass_all"]:
        print("If flat on pi05_libero, see CLAUDE.md §9: fall back to pi05_base / pi05_droid before RL.")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    run_audit(tyro.cli(Args))
