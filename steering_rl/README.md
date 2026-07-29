# RL over the Steering Interface of a Frozen π0.5 — Results

**TL;DR (measured, not invented).** We asked whether a small policy that *chooses the
free-text instruction* handed to a **frozen** π0.5 VLA can beat the best fixed
instruction on multi-stage LIBERO tasks. Across an 8-task steerability audit plus a
checkpoint fallback, the answer on the available frozen checkpoints is **no**: the
LIBERO-competent checkpoint (`pi05_libero`) is *prompt-robust* — it does the task
from visual context and largely ignores the instruction — so there is essentially no
headroom for a learned steering policy to beat a fixed prompt. The
better-language-following checkpoint (`pi05_base`) **cannot do LIBERO** (0/30
zero-shot). This is a well-evidenced **negative result**, delivered with a reusable
steerability-audit harness.

> Numbers here are measured on the cloud-served frozen policy and are reproducible
> with the harness below. Nothing in this project fine-tunes or backprops the VLA.

---

## 1. What we set out to test

- **Frozen VLA as environment dynamics.** Take the pretrained `pi05_libero` π0.5
  checkpoint, never compute a gradient through it, and treat it as part of a
  semi-MDP. Steering happens *only* through the free-text `prompt` string handed to
  it at each replan boundary (the prompt is the action).
- **Hypothesis.** A state-dependent policy that *chooses the instruction* can beat any
  single fixed instruction on multi-stage tasks.
- **The gate before any RL (Phase C).** For the hypothesis to have a control surface,
  swapping the prompt (seed + env fixed) must (a) produce a meaningful spread in
  success across legitimate phrasings, and (b) make correct instructions beat
  misleading ones. If the prompt barely moves the outcome, there is nothing to learn.

## 2. Architecture (the freeze mechanism)

The freeze is enforced by openpi's **policy-server / client boundary**: the frozen
JAX checkpoint is served unmodified (`scripts/serve_policy.py --env LIBERO`), and our
client sends observations with a chosen `prompt` string over a websocket. The prompt
travels as a raw string and is tokenized *server-side*, so per-call instruction
injection needs zero openpi edits. All of our code lives in `steering_rl/`; openpi is
a frozen dependency.

Two ways to run the audit are provided:
- **All-Modal** (`steering_rl/modal/`): server + LIBERO client co-located in one Modal
  GPU container. Efficient for eval sweeps.
- **Local split** (`steering_rl/local/`): LIBERO rollouts + rendering run locally
  (validated on an RTX 4070 via `MUJOCO_GL=glx`); only π0.5 inference runs on Modal,
  reached over a tunnel. Free rendering, but per-inference network latency makes it
  slower than co-located for sequential sweeps — its value is free RL-loop development
  against the fake env, and (future) parallel rollouts.

## 3. Evaluation protocol

- **Tasks:** multi-stage LIBERO-10 (`libero_10`) pick-and-place / articulated tasks.
- **Success:** the env's own BDDL goal-predicate satisfaction (`done`) — nothing else.
  Reward/eval never come from the injected prompt.
- **Menus:** per task, ~8–10 prompts in 4 categories — canonical (the exact string
  LIBERO feeds), paraphrases, decomposed subgoals, and misleading (another task's
  instruction).
- **Seeds/episodes:** fixed init-state indices reused across all prompts. Because of a
  Modal compute budget, the runs reported here use a **reduced** episode count (10 for
  the locked set, 6 for candidates) rather than the full 20 — internally consistent,
  but noisier than the intended protocol. This does not change the qualitative
  conclusion (see §7).

## 4. Result A — `pi05_libero` is competent but prompt-robust

Locked 5 tasks, 10 episodes/prompt. "Headroom" = does any prompt *beat* the canonical
one (the only way a learned policy could beat best-fixed)?

| task | canonical | best prompt | misleading | spread | gate | headroom |
|---|---|---|---|---|---|---|
| soup + tomato → basket | 1.00 | 1.00 | 0.00 | 0.20 | pass | none (ceiling) |
| soup + cream-cheese → basket | 1.00 | 1.00 | 0.80 | 0.00 | fail | none |
| cream-cheese + butter → basket | 1.00 | 1.00 | 0.25 | 0.00 | fail | none |
| black-bowl → drawer + close | 0.90 | 1.00 | **0.95** | 0.10 | fail | none (prompt-invariant) |
| mug → microwave + close | 0.90 | 1.00 | 0.40 | 0.30 | pass | small (0.90→1.00) |

Candidate tasks screened afterward (6 episodes/prompt), chosen for
"which-object-goes-where" disambiguation structure:

| task | canonical | best prompt | crossed-misleading | spread | gate | headroom |
|---|---|---|---|---|---|---|
| white-mug→left, y/w-mug→right | 1.00 | 1.00 | **1.00** | 0.00 | fail | none (prompt-invariant) |
| white-mug→plate, pudding→right | 0.83 | 1.00 | 0.92 | 0.83 | pass | small (0.83→1.00) |
| turn-on-stove + moka-pot | 1.00 | 1.00 | 0.83 | 0.17 | pass | none (ceiling) |

**Reading:** 4/8 tasks "pass" the spread gate, but on the stricter test that the
project actually needs — a prompt that *beats* canonical — only **2/8** have any
headroom, and even there the best *fixed* prompt is already ~1.0, leaving no room for
a *learned* policy to beat best-fixed.

## 5. Result B — the misleading→failure signal is a confound

Early on, misleading prompts drove success to 0 on soup+tomato, which looked like
strong instruction-following. It is not. Those misleading prompts named objects
**absent** from the scene (a bowl/drawer that isn't there), so the policy simply
failed to act. When the misleading prompt uses **present** objects with a swapped
target (the "crossed-destination" prompt on the 2-mug/2-plate task), the policy
**ignores the swap and places correctly anyway** (1.00). So the frozen policy is
driven by visual scene context, not the instruction — it is prompt-*robust*, not
prompt-*following*.

## 6. Result C — the `pi05_base` fallback is a dead end

`pi05_libero` *is* `pi05_base` fine-tuned on LIBERO, so `pi05_base` is the natural
"better language-following, lower competence = more headroom" fallback. We served
`pi05_base` params with LIBERO norm stats + transforms
(`steering_rl/serving/serve_base_libero.py`, wrapping openpi's public API). On
soup+tomato (which `pi05_libero` aces): **0/30 success**, every episode ran the full
530 steps with no errors — the robot moved but never completed the task. `pi05_base`
is genuinely incompetent at LIBERO zero-shot (never trained on this embodiment; its
state normalization also differs). `pi05_droid` is a different embodiment entirely.
Fine-tuning any of them is out of scope (frozen-VLA invariant).

## 7. Conclusion

On the available **frozen** π0.5 checkpoints, learned prompt-steering has **no viable
headroom on LIBERO**: the competent checkpoint is prompt-robust, and the
language-following checkpoints cannot do the tasks. The core hypothesis cannot be
demonstrated without either (a) a checkpoint that is both LIBERO-competent *and*
prompt-sensitive — which does not exist among the released frozen checkpoints — or
(b) relaxing the frozen-VLA invariant to fine-tune one, which the project forbids.

This is a legitimate negative result. It is also a useful one: it says that steering a
*strong, well-trained* VLA through its text interface buys little, because such a model
has already learned to solve the task from perception and does not need the
instruction to disambiguate.

## 8. What's released (the harness)

- `steering_rl/libero/` — locked tasks + candidates (`tasks.py`), prompt menus
  (`menus.py`), BDDL predicate-vector wrapper (`predicates.py`), the wrapped episode
  loop (`episode.py`), the resumable audit sweep + gate aggregation (`run_audit.py`),
  and a readable report renderer (`report.py`).
- `steering_rl/modal/` — one-container all-Modal audit + a `serve` tunnel; `run_audit`
  supports `--candidates` and a `base_model` flag.
- `steering_rl/serving/serve_base_libero.py` — the `pi05_base`-on-LIBERO server.
- `steering_rl/local/` — local-render + remote-inference orchestration.
- `steering_rl/{fake_env,policy,ppo}.py` — the Phase-A PPO machinery, validated on a
  fake env (kept for the RL path if the frozen constraint is ever relaxed).
- Measured results: `steering_rl/results/`.

### Reproduce

```bash
# plumbing (no GPU/MuJoCo):
uv run python -m pytest steering_rl/

# audit on Modal (after `modal setup`), from the repo root:
modal run steering_rl/modal/audit.py --episodes 3 --tasks-limit 1     # smoke
modal run steering_rl/modal/audit.py                                  # locked set
# candidate screen / base fallback are spawned via run_audit (--candidates / base_model=True)

# render any pulled audit_records.jsonl into a gate report:
uv run python -m steering_rl.libero.report data/steering_rl/audit/audit_records.jsonl
```

## 9. Limitations & future work

- **Reduced episode counts** (10/6 vs. the intended 20) from a compute budget; rates
  are noisier, though the ceiling/robustness pattern is consistent across 8 tasks.
- **Fixed-prompt audit is a lower bound** on state-dependent steering; a per-stage
  switching policy could in principle exploit structure a fixed prompt can't. But with
  best-fixed already at ceiling on the tasks with any headroom, the room for it to win
  is small.
- **Future work** (would require relaxing the frozen invariant or new checkpoints):
  fine-tune `pi05_base` on LIBERO *with prompt variation* to create a competent,
  language-following policy, then test learned steering against it; or evaluate under
  harder conditions (distractors, tougher init states) that push a strong policy below
  ceiling.
