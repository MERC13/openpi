"""Steering-RL: an RL policy over the prompt interface of a frozen pi0.5 VLA.

See CLAUDE.md at the repo root for the full project brief. This package holds
ONLY our code; openpi is treated as a frozen dependency and is never edited.
Phase A (this milestone) is a fully local, CPU-friendly validation of the PPO
machinery against a fake env, with zero pi0.5 / MuJoCo / GPU dependency.
"""
