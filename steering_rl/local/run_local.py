"""Run a LOCAL LIBERO audit/rollout against the Modal-hosted frozen pi0.5.

The frugal split (CLAUDE.md §3): LIBERO physics + rendering (MUJOCO_GL=glx on the
local GPU) and the whole audit/RL loop run locally for free; only pi0.5 inference
runs on Modal, reached over a public TCP tunnel. This script:

  1. reads the Modal ``serve`` function's tunnel address from the ``steering-tunnel``
     Dict (spawning ``serve`` if nothing is up),
  2. writes the LIBERO config.yaml pointing at this repo's third_party/libero,
  3. runs ``steering_rl/libero/run_audit.py`` in the LOCAL libero venv, pointed at
     the tunnel.

Run under a python that has ``modal`` installed (the modal CLI venv), from the
repo root:

    ~/.local/share/uv/tools/modal/bin/python steering_rl/local/run_local.py \
        --client-args "--candidates --episodes 8 --out-dir data/steering_rl/audit_candidates"

Env overrides: STEERING_LIBERO_VENV (default ~/steering-libero-venv),
STEERING_LIBERO_CONFIG (default ~/.config/steering-libero).
"""

import argparse
import os
import pathlib
import subprocess
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBERO_VENV = pathlib.Path(os.environ.get("STEERING_LIBERO_VENV", os.path.expanduser("~/steering-libero-venv")))
CONFIG_DIR = pathlib.Path(os.environ.get("STEERING_LIBERO_CONFIG", os.path.expanduser("~/.config/steering-libero")))
APP_NAME = "steering-rl-audit"


def write_libero_config() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = REPO / "third_party/libero/libero/libero"
    (CONFIG_DIR / "config.yaml").write_text(
        f"benchmark_root: {base}\n"
        f"bddl_files: {base}/bddl_files\n"
        f"init_states: {base}/init_files\n"
        f"datasets: {REPO}/third_party/libero/libero/datasets\n"
        f"assets: {base}/assets\n"
    )


def get_tunnel(spawn_minutes: float) -> dict:
    info = modal.Dict.from_name("steering-tunnel", create_if_missing=True)
    addr = info.get("address")
    if addr is None:
        print("[local] no server tunnel published; spawning the Modal serve function ...")
        modal.Function.from_name(APP_NAME, "serve").spawn(minutes=spawn_minutes)
        for _ in range(180):  # up to ~15 min for cold start + checkpoint restore
            addr = info.get("address")
            if addr:
                break
            time.sleep(5)
    if not addr:
        raise SystemExit("serve did not publish a tunnel address in time")
    print(f"[local] frozen pi0.5 tunnel: ws://{addr['host']}:{addr['port']}")
    return addr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client-args", default="--candidates --episodes 8", help="forwarded to run_audit.py")
    ap.add_argument("--serve-minutes", type=float, default=120.0, help="serve lifetime if this spawns it")
    ap.add_argument("--mujoco-gl", default="glx", help="local render backend (glx=GPU via WSLg, or osmesa)")
    args = ap.parse_args()

    write_libero_config()
    addr = get_tunnel(args.serve_minutes)

    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO}:{REPO}/packages/openpi-client/src:{REPO}/third_party/libero",
        "LIBERO_CONFIG_PATH": str(CONFIG_DIR),
        "MUJOCO_GL": args.mujoco_gl,
        "PYOPENGL_PLATFORM": args.mujoco_gl,
    }
    cmd = [
        str(LIBERO_VENV / "bin/python"),
        "steering_rl/libero/run_audit.py",
        "--host", f"ws://{addr['host']}",
        "--port", str(addr["port"]),
        *args.client_args.split(),
    ]
    print("[local] running:", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
