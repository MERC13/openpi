Notice: This is the local codebase.

# CLAUDE.md — RL over the Steering Interface of a Frozen π0.5

This file briefs Claude Code on a research project built **on top of** the openpi
repository (Physical Intelligence). Read it fully before acting. It defines what
is frozen, the hardware constraints, the eval protocol, and the hard rules. When
in doubt, prefer the rule in this file over a locally convenient shortcut, and
ask before doing anything this file forbids.

---

## 1. One-paragraph project statement

We freeze a pretrained π0.5 VLA (`pi05_libero` checkpoint from openpi) and never
compute a gradient through it. We train a small (~1M-param) high-level RL policy
whose **action space is the free-text `prompt` string** handed to π0.5 at each
replan boundary. The hypothesis: a state-dependent policy that *chooses the
instruction* can beat any fixed instruction on multi-stage LIBERO tasks. This is
a semi-MDP where the frozen VLA is part of the environment dynamics. The
contribution is the comparison **learned steering vs. best fixed prompt** — not a
training curve.

## 2. Non-negotiable invariants (violating any of these breaks the project)

- **π0.5 is FROZEN.** No fine-tuning, no LoRA, no gradient through the VLA, ever.
  If a task seems to need fine-tuning, STOP and flag it — the answer is almost
  always "use the policy-server boundary instead."
- **The RL policy is the only thing that trains.** It is a small network
  (~1M params) over [CLIP image embedding ‖ LIBERO predicate vector].
- **Reward comes from LIBERO's own goal predicates, NEVER from the injected
  prompt.** The model can be told to do task B while the env checks task A; only
  the env's success check counts. (This is the openpi issue #446 trap.)
- **The prompt string is the action.** All steering happens by choosing which
  text goes into the `prompt` field before tokenization. Do not modify images,
  state, or any other model input as a control channel.
- **Eval numbers are measured, never invented.** If asked for a success rate,
  run the eval; do not fill in a plausible number from memory or from the paper.

## 3. Hardware & environment constraints (these shape every decision)

- **Local dev machine: laptop RTX 4070, 8GB VRAM, Windows + WSL2 (Ubuntu).**
  8GB is BELOW π0.5's >8GB inference floor once WDDM overhead is counted.
  **Therefore π0.5 does NOT run locally.** Do not attempt to load the full
  checkpoint on the local card.
- **Division of labor (this is the core architecture):**
  - *Local (WSL2, 4070):* everything EXCEPT π0.5 inference — the PPO loop, the
    policy net, the CLIP encoder, reward logic, the fake-env harness, all
    analysis and plotting. This must run without any π0.5 or MuJoCo dependency.
  - *Free cloud (Lightning AI / Kaggle T4 16GB):* the frozen π0.5 policy SERVER
    and LIBERO rollout collection, via openpi's Docker LIBERO flow.
  - The two halves talk over openpi's existing websocket policy-server/client
    split. The client (our RL code) can run locally and point at a remote server.
- **Prefer openpi's Docker `compose.yml` for anything touching MuJoCo/LIBERO.**
  Do not hand-assemble the EGL/OpenGL rendering stack unless Docker is
  impossible. If EGL fails, the documented fallback is `MUJOCO_GL=glx`.
- **Frozen server runs the stock JAX checkpoint — do NOT convert π0.5 to PyTorch.**
  Because we never train or backprop the VLA, the PyTorch VLA path buys us nothing.
  Serve the native JAX checkpoint straight from `gs://openpi-assets` with
  `scripts/serve_policy.py --env LIBERO` (the LIBERO default is
  `Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero")`,
  a JAX `params/` dir). `policy_config.create_trained_policy` auto-detects framework
  by the presence of `model.safetensors`; absent it → JAX. **Skip the
  `transformers_replace/` copy and the JAX→PyTorch conversion entirely** — they only
  matter for a PyTorch VLA we don't need, and the copy mutates the uv cache.
- **PyTorch is for OUR code only** — the ~1M-param RL net, the CLIP encoder, PPO.
  It never loads the VLA. This runs locally on the 4070 with no `transformers_replace`
  patch and no π0.5/MuJoCo dependency.

## 4. Repository orientation (openpi — do not edit these, wrap them)

Our code lives in a NEW top-level directory `steering_rl/` (create it). We treat
openpi as a frozen dependency. Key openpi files to READ (not modify):

- `src/openpi/training/config.py` — the config registry. `pi05_libero` (line ~743) is
  our target config: `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`,
  `prompt_from_task=True`. (`discrete_state_input` defaults to `pi05`=True but this config
  overrides it back to False.) `prompt_from_task` is a **train-time** dataset transform
  (`data_loader.py`, needs `task_index`); it is ABSENT from the serving pipeline, so it does
  NOT clobber a client-supplied prompt at inference.
- `src/openpi/policies/libero_policy.py` — `LiberoInputs`/`LiberoOutputs`.
  `LiberoInputs` passes `prompt` through as a **raw string** (line ~80); it does NOT
  tokenize. Tokenization happens SERVER-SIDE in `transforms.TokenizePrompt` (after the
  websocket boundary). **This is why client-side injection just works with zero openpi
  edits:** put a different string in `obs["prompt"]` on each `client.infer()` call and
  the server tokenizes the new string every time. `InjectDefaultPrompt` only fills a
  prompt when the client omits one, so a client-supplied string always wins.
  The Libero obs dict schema (what the client must send):
  `observation/image` (uint8 HWC), `observation/wrist_image` (uint8 HWC),
  `observation/state` (8-dim: eef pos 3 ‖ axis-angle 3 ‖ gripper 2), `prompt` (str).
- `examples/libero/main.py` — the episode loop, `replan_steps`, success check.
  Our RL client is a wrapped version of this loop. Note: it calls `client.infer`
  directly with a manual `collections.deque` replan buffer (`replan_steps` default **5**;
  the action chunk is `action_horizon`=10 but only the first 5 are executed before the
  next `infer`) — it does NOT use `ActionChunkBroker`. The steering decision point is
  each time the deque empties. The third-person frame to CLIP-encode is
  `obs["agentview_image"]` (raw 256×256, stored upside-down — main.py flips it 180° for
  the model; use the upright frame for CLIP).
- `scripts/serve_policy.py` — the frozen policy server. We run this unmodified on
  the cloud box.
- `packages/openpi-client/` — the websocket client we build our RL wrapper around.

**Rule: never edit files under `src/openpi/`, `scripts/`, or `examples/` in place.**
Copy or wrap. All our logic goes in `steering_rl/`.

## 5. The plan, in phases (reference — build in this order)

Each phase has a hard exit criterion. Do not start a phase before the previous
one's exit criterion is met. Announce which phase you're in.

**Phase A — Local RL machinery against a FAKE env (no π0.5, no MuJoCo).**
Build in `steering_rl/`:
- `fake_env.py`: a mock LIBERO-like env returning dummy image + predicate vector,
  with a toy reward where a known "correct" instruction sequence wins. This
  exists so PPO can be validated with zero GPU-heavy deps.
- `policy.py`: 2-layer MLP over [CLIP-dim ‖ predicate-dim] → categorical over an
  N-item instruction menu. (Stub CLIP with a random projection locally.)
- `ppo.py`: PPO + GAE. This is a port of standard CS-285-style PPO; keep it
  readable and unit-testable.
- Exit criterion: PPO provably learns the toy task on the fake env (reward rises
  to near-optimal), running entirely on the local 4070 or even CPU.

**Phase B — Cloud: stand up frozen π0.5 + one LIBERO episode.**
- Get `pi05_libero` serving via `serve_policy.py` in the Docker LIBERO flow.
- Run ONE episode of ONE LIBERO-10 task end to end.
- Exit criterion: one episode produces an action chunk and a success/fail. Do NOT
  attempt full-suite reproduction of the 92.4 number — that is explicitly out of
  scope (see §6).

**Phase C — Instruction menu + steerability audit (THE GATE).**
- Pick 5 multi-stage LIBERO-10 tasks (pick-and-place-into-container types, which
  have the most headroom below the 92.4 ceiling).
- Per task, build ~10 candidate prompts: canonical string, 3–5 LLM paraphrases,
  2–4 decomposed sub-instructions.
- Audit: holding seed+env fixed, swap the prompt and measure success spread.
  Require (a) meaningful spread and (b) correct-instruction > misleading-instruction.
- Exit criterion: the audit shows the prompt is a usable control surface. A flat result
  on `pi05_libero` is a LIKELY outcome, not an edge case (see §9) — it was trained on one
  prompt per task. If flat, FLAG IT and audit `pi05_base`/`pi05_droid` (stronger language
  following) before any RL. Do not proceed to Phase D on a flat interface.
- **How to run (cloud, built):** the harness lives in `steering_rl/libero/`
  (locked tasks in `tasks.py`, prompt menus in `menus.py`, predicate wrapper in
  `predicates.py`, sweep in `run_audit.py`). From the openpi repo root on the cloud box:
  ```
  # smoke-test the pipeline first (3 episodes/prompt), then scale to the full 20
  SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--host 0.0.0.0 --port 8000 --episodes 3" \
    docker compose -f steering_rl/docker/compose.audit.yml up --build
  # full audit: drop `--episodes 3` (defaults to 20 per §7)
  ```
  Output → `data/steering_rl/audit/`: `audit_records.jsonl` (resumable — reruns
  skip finished task/prompt/episode triples) and `audit_summary.json`; the run
  prints per-task spread + the gate verdict. Full sweep ≈ 5×10×20 ≈ 1000 episodes.
  Run `uv run python -m pytest steering_rl/libero/` locally to check the plumbing
  (no cloud/MuJoCo needed) before launching.

**Phase D — Real RL: connect Phase-A machinery to the frozen server.**
- Replace `fake_env` with the real LIBERO client → frozen π0.5 server.
- State: real CLIP(ViT-B) embedding of current frame ‖ real BDDL predicate vector.
  **The predicate vector is NOT exposed by the env's public API** (only terminal `done`
  is — `done` == goal satisfaction, `bddl_base_domain.py`). Compute it yourself with a
  thin wrapper reaching the underlying problem env:
  `[float(prob._eval_predicate(s)) for s in prob.parsed_problem["goal_state"]]`
  where `prob = env.env`. Two caveats that shape the design:
  (a) `goal_state` length **varies per task** → a fixed-width RL state needs per-task
  schema or padding to max arity;
  (b) predicates are noisy binary geometric checks (contact-based `On`/`In`) that can
  **flicker / regress** step-to-step.
- Action: categorical over the task's menu, incl. a "canonical string" default arm.
- Decision points: every `replan_steps` boundary (default 5 env steps).
- Reward: terminal success from LIBERO goal check + small shaping per newly
  satisfied predicate (only if sparse reward learns too slowly). Because predicates
  flicker, shape on the **running max** of satisfied-count, not raw step-to-step deltas.
- Run parallel envs against one batched server; checkpoint + resume aggressively
  (free-tier sessions time out).
- Exit criterion: held-out success rate plateaus.

**Phase E — The four-bar comparison (THE RESULT — never cut).**
On the SAME 5 tasks, SAME 20 episodes, SAME seeds as the audit, measure:
1. canonical prompt (baseline)  2. random menu selection
3. best-fixed prompt per task (brute-force menu, no RL)  4. learned RL policy
- The claim = **(4) beats (3)**. Beating (1)/(2) proves nothing on its own.
- Report per-task (not just averaged) so ceiling effects are visible, with error
  bars across seeds.

**Phase F — One analysis + writeup.**
- Categorize the canonical policy's failures (wrong object / premature grasp /
  stall / wrong ordering) and show which categories learned steering repairs.
- Assemble repo + short writeup. Release the audit harness as a usable tool.

## 6. Explicitly OUT of scope (do not build these without being asked)

- Fine-tuning / LoRA of π0.5 (Phase-0 conditioning fine-tune from earlier drafts
  is CUT).
- Full 4-suite LIBERO reproduction or matching the published 92.4 number. We use
  a 5-task, 20-episode, internally-consistent subset instead.
- Subgoal-image and metadata conditioning channels (text prompt only).
- The bandit tier (go straight to PPO).
- Off-policy comparison, cross-model transfer, subgoal-hacking probe. All are
  "future work," not v1.

## 7. Evaluation protocol (fix this once, reuse everywhere)

- Task subset: 5 chosen multi-stage LIBERO-10 tasks, fixed at Phase C.
- Episodes: 20 per task per condition. Seeds: a fixed list, recorded and reused
  verbatim across ALL conditions (audit, baselines, RL eval).
- Success: LIBERO BDDL goal predicate satisfaction, from the env — nothing else.
- Every comparison is internally consistent (same tasks/episodes/seeds/harness).
  Internal consistency, not agreement with the paper, is what validates results.
- Report per-task numbers + averages + across-seed error bars.

## 8. Working style for Claude Code

- Announce the current phase and its exit criterion at the start of a work block.
- Keep openpi untouched; put everything in `steering_rl/`.
- Write small, testable modules; unit-test `ppo.py` and `fake_env.py` before
  they ever touch π0.5.
- Never invent a metric. If a number is needed and no eval has produced it, say
  so and run (or propose) the eval.
- When a step needs the cloud GPU, make the code cloud-ready but don't assume the
  local machine can run π0.5 — guard π0.5 imports so local dev never hard-depends
  on them.
- Prefer the policy-server/client boundary as the freeze mechanism; if you find
  yourself wanting a gradient through π0.5, stop — that's a design error.
- Flag, don't guess, when repo reality (file paths, config fields, the LIBERO
  loop) differs from what this file describes — the repo is ground truth; this
  file is intent.

## 9. Known risks & open questions (READ before Phase C/D)

Two risks are load-bearing enough to plan around explicitly. Neither changes the
headline plan; both change how much we should bet on the default path.

- **Prompt-invariance of `pi05_libero` (PRIMARY RISK — this is what Phase C gates).**
  `pi05_libero` was fine-tuned with `prompt_from_task=True`, i.e. it saw exactly ONE
  prompt per task in training (the ground-truth string). It may have learned to barely
  attend to the prompt, since prompt never had to disambiguate anything. If so, the
  Phase-C audit comes back **flat** and the core hypothesis has no control surface on
  this checkpoint. **Treat the `pi05_base` / `pi05_droid` fallback as a LIKELY path, not
  a remote contingency** — the README calls out `pi05_droid` as having good
  language-following; `pi05_libero` is not described that way. Budget audit effort across
  all three checkpoints, not just `pi05_libero`.

- **PPO sample cost vs. a tiny discrete action space (secondary).** Rollouts go through
  an expensive VLA on a free-tier T4 with session timeouts; each episode has only ~tens
  of decision points and the menu is ~10 arms. PPO's credit-assignment advantage may not
  pay for its sample cost at this scale. A **contextual bandit (state → arm, terminal
  reward) is a sanctioned pragmatic first rung** — it validates the state features and the
  menu with far fewer samples, and PPO remains the goal (§5 Phase D) once steering is
  shown to help. This is a deliberate softening of §6's "bandit cut": don't build the
  bandit speculatively, but reach for it before assuming slow PPO convergence is a bug.

## 10. Verified repo facts (anchors — so future sessions don't re-derive)

- Prompt is a raw string client→server; tokenized server-side in
  `src/openpi/transforms.py` `TokenizePrompt`. `LiberoInputs` passes it through
  (`src/openpi/policies/libero_policy.py`). `InjectDefaultPrompt` only fills when absent.
- `pi05_libero` config: `src/openpi/training/config.py` (~L743),
  `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`.
- Serve JAX checkpoint: `scripts/serve_policy.py --env LIBERO`; framework auto-detect in
  `src/openpi/policies/policy_config.py` (`model.safetensors` → PyTorch, else JAX).
- LIBERO loop / success / replan: `examples/libero/main.py` (gym 4-tuple, `done`==success,
  manual `replan_steps` deque, seed default 7, init states via `get_task_init_states`).
- Per-predicate progress: `third_party/libero/.../bddl_base_domain.py` +
  problem files' `_eval_predicate` over `parsed_problem["goal_state"]`.
- LIBERO-10 == `libero_10` == `libero_long`; task strings parsed from BDDL filenames in
  `third_party/libero/libero/libero/benchmark/__init__.py`.