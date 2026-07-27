"""Modal deployment of the Phase C steerability audit.

This is the cloud path when Docker/free-tier GPUs are unavailable (see the
sibling ``steering_rl/docker/`` compose flow). It packages the *same* two
runtimes the compose flow uses — the frozen JAX pi0.5 server and the Python-3.8
LIBERO audit client — into one Modal GPU container, and runs the identical
``steering_rl/libero/run_audit.py`` sweep. No openpi files are edited (CLAUDE.md
§4); nothing here trains or backprops the VLA (CLAUDE.md §2).
"""
