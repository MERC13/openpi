"""LIBERO-side code for the steering project (Phases C-E).

Everything here runs against the cloud-served frozen pi0.5 (via openpi's
websocket policy server) inside the Docker LIBERO flow. Heavy deps (libero,
MuJoCo, openpi_client) are imported lazily inside functions so that the pure
logic modules (tasks, menus, predicates, audit aggregation) remain importable
and unit-testable locally with no GPU / MuJoCo dependency.
"""
