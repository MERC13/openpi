"""One LIBERO episode against the frozen pi0.5 server, with prompt steering.

A wrapped adaptation of ``examples/libero/main.py`` (openpi's stock loop, left
untouched per the no-edit rule). The one behavioural change is the injection
point: instead of always sending ``task.language`` as the ``prompt``, this loop
asks a ``prompt_provider`` for the prompt at each replan boundary. That single
hook covers both:

- **Phase C audit**: pass a fixed string -> the same prompt every replan.
- **Phase D RL**: pass a callable ``ctx -> str`` -> the RL policy picks a menu
  item each replan from the current [predicate vector ‖ image] context.

libero / openpi_client / MuJoCo are imported lazily inside the functions so this
module imports cleanly on a machine with none of them (CLAUDE.md §8).
"""

from __future__ import annotations  # 3.8 client venv: keep PEP 585/604 generics lazy

import collections
import dataclasses
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from steering_rl.libero import predicates
from steering_rl.libero import tasks

if TYPE_CHECKING:
    from collections.abc import Callable

    # A prompt provider is either a fixed string or a callable over a step context.
    # Referenced only in (lazy, stringized) annotations, so it need not exist at runtime.
    PromptProvider = str | Callable[[dict], str]

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class EpisodeResult:
    success: bool
    steps: int
    num_replans: int
    prompts_used: list[str]
    predicate_trajectory: list[np.ndarray]  # predicate vector sampled at each replan
    final_predicates: np.ndarray | None
    error: str | None = None


def make_env(task_name: str, *, resolution: int = LIBERO_ENV_RESOLUTION, seed: int = tasks.ENV_SEED):
    """Build the LIBERO env + benchmark task for ``task_name`` (lazy libero import)."""
    import pathlib

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[tasks.TASK_SUITE_NAME]()
    task_id = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == task_name)
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task, initial_states


def _resolve_prompt(provider: PromptProvider, ctx: dict) -> str:
    return provider(ctx) if callable(provider) else provider


def _quat2axisangle(quat):
    """Copied from robosuite (via examples/libero/main.py)."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _preprocess_images(obs, resize_size: int):
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    return img, wrist


def run_episode(
    client,
    env,
    initial_state,
    prompt_provider: PromptProvider,
    *,
    max_steps: int = tasks.MAX_STEPS,
    replan_steps: int = 5,
    num_steps_wait: int = tasks.NUM_STEPS_WAIT,
    resize_size: int = 224,
    record_predicates: bool = True,
) -> EpisodeResult:
    """Run one episode from a fixed ``initial_state``; steer the prompt per replan.

    Success is the env's own terminal ``done`` (== BDDL goal satisfaction).
    """
    env.reset()
    action_plan: collections.deque = collections.deque()
    obs = env.set_init_state(initial_state)

    problem_env = None
    if record_predicates:
        try:
            problem_env = predicates.find_problem_env(env)
        except AttributeError as exc:  # pragma: no cover - depends on libero layout
            logger.warning("predicate logging disabled: %s", exc)

    prompts_used: list[str] = []
    predicate_trajectory: list[np.ndarray] = []
    num_replans = 0
    done = False
    success = False
    error = None

    t = 0
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img, wrist = _preprocess_images(obs, resize_size)

            if not action_plan:
                pred_vec = predicates.predicate_vector(problem_env) if problem_env is not None else None
                ctx = {
                    "step": t,
                    "replan_index": num_replans,
                    "predicate_vector": pred_vec,
                    "agentview_image": obs["agentview_image"],
                }
                prompt = str(_resolve_prompt(prompt_provider, ctx))

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": prompt,
                }
                action_chunk = client.infer(element)["actions"]
                if len(action_chunk) < replan_steps:
                    raise ValueError(
                        f"policy returned {len(action_chunk)} actions, need >= replan_steps={replan_steps}"
                    )
                action_plan.extend(action_chunk[:replan_steps])

                prompts_used.append(prompt)
                if pred_vec is not None:
                    predicate_trajectory.append(pred_vec)
                num_replans += 1

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as exc:  # match main.py: a sim/inference error ends the episode
        error = f"{type(exc).__name__}: {exc}"
        logger.error("episode ended on exception: %s", error)

    final_pred = predicates.predicate_vector(problem_env) if problem_env is not None else None
    return EpisodeResult(
        success=success,
        steps=t,
        num_replans=num_replans,
        prompts_used=prompts_used,
        predicate_trajectory=predicate_trajectory,
        final_predicates=final_pred,
        error=error,
    )
