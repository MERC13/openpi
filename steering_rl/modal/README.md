# Phase C audit on Modal (no Docker / free-tier GPU alternative)

Runs the **same** Phase C steerability audit as `steering_rl/docker/compose.audit.yml`,
but on an on-demand [Modal](https://modal.com) GPU instead of a long-lived cloud box.
Use this when the Docker/free-tier GPU path is unavailable.

One Modal GPU container holds **both** runtimes the audit needs (they have
conflicting Python/torch/JAX pins, so they live in separate uv venvs and share the
GPU):

| runtime | venv | role |
|---|---|---|
| frozen **JAX** pi0.5 server | `/server_venv` (Py 3.11) | background subprocess, `scripts/serve_policy.py --env LIBERO` |
| **LIBERO** audit client | `/.venv` (Py 3.8, MuJoCo + torch-cu113) | foreground, `steering_rl/libero/run_audit.py` |

The client talks to the server over `localhost:8000`. Nothing here edits openpi
(CLAUDE.md §4) or trains/backprops the VLA (CLAUDE.md §2). Success = the env's own
BDDL `done` — measured, never invented.

## One-time setup

```bash
uv tool install modal          # or: pip install modal
modal setup                    # opens a browser to auth your Modal account
```

Two persistent Modal Volumes are auto-created on first run:
- `openpi-assets` — caches the `gs://openpi-assets` pi0.5 checkpoint so it downloads once.
- `steering-audit` — holds the resumable audit log (`/audit/audit_records.jsonl` + `audit_summary.json`).

## Run it

**Always from the openpi repo root** (the image bakes repo dirs by relative path):

```bash
# smoke test first: 3 episodes/prompt, only the first locked task, cheapest GPU
modal run steering_rl/modal/audit.py --episodes 3 --tasks-limit 1

# full audit: 20 episodes/prompt across all 5 locked tasks (~1000 episodes)
modal run steering_rl/modal/audit.py
```

The first run builds the image (both venvs — several minutes, cached afterward) and
downloads the checkpoint. The gate summary prints at the end. Runs are **resumable**:
a re-run skips finished `(task, prompt, episode)` triples, so a timeout loses at most
one episode.

Pull the results locally:

```bash
modal volume get steering-audit /audit ./data/steering_rl/audit
```

## Notes

- GPU is `T4` (16GB) by default — pi0.5's >8GB floor fits. Bump `GPU = "A10G"` in
  `audit.py` if you hit OOM.
- The audit interprets its own output against the Phase C gate (CLAUDE.md §5): a flat
  result on `pi05_libero` is a **likely** outcome (§9); if flat, the fallback is to audit
  `pi05_base` / `pi05_droid` before any RL — change the served checkpoint via the server's
  config, not by editing openpi.
- Plumbing check before launching (local, no GPU/MuJoCo):
  `uv run python -m pytest steering_rl/libero/`.
