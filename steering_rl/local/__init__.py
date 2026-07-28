"""Local-side orchestration for the frugal split architecture (CLAUDE.md §3).

LIBERO rollouts + rendering (MUJOCO_GL=glx on the local GPU) + CLIP + the RL loop
run locally for free; only the frozen pi0.5 inference runs on Modal, reached over
a public tunnel. This keeps the expensive VLA on the cloud while the cheap,
render-heavy work stays on the 4070.
"""
