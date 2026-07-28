"""Run the Phase C steerability audit on Modal (cloud GPU), no Docker required.

Why this exists
---------------
The compose flow in ``steering_rl/docker/`` needs a long-lived cloud GPU box. When
that is unavailable (free-tier exhausted), Modal gives us on-demand GPU with a
per-second bill and a persistent Volume for the resumable audit log. This module
reproduces the *exact* two-runtime split of ``compose.audit.yml`` inside a single
Modal GPU container:

  * the frozen **JAX** pi0.5 server (Python 3.11 venv, ``/server_venv``) runs as a
    background subprocess — ``scripts/serve_policy.py --env LIBERO`` (CLAUDE.md §3:
    stock JAX checkpoint from gs://openpi-assets, never converted to PyTorch);
  * the **LIBERO** audit client (Python 3.8 venv, ``/.venv``, MuJoCo + torch-cu113)
    runs in the foreground — ``steering_rl/libero/run_audit.py`` — talking to the
    server over ``localhost:8000``.

The two environments have conflicting Python/torch/JAX pins, so they stay in
separate uv venvs inside one image and share the single GPU (JAX inference +
EGL rendering both fit on a T4). No openpi source is edited (CLAUDE.md §4); the
VLA is never trained or backpropped (CLAUDE.md §2). Success = the env's own BDDL
``done`` (CLAUDE.md §2/§7) — measured on the frozen server, never invented.

Run it (after ``modal setup``)
------------------------------
    # smoke test: 3 episodes/prompt, one task, cheapest GPU
    modal run steering_rl/modal/audit.py --episodes 3 --tasks-limit 1

    # full audit: 20 episodes/prompt across all 5 locked tasks (~1000 episodes)
    modal run steering_rl/modal/audit.py

Output lands on the ``steering-audit`` Modal Volume under ``/audit`` (resumable —
re-running skips finished (task, prompt, episode) triples, so a timeout loses at
most one episode). Pull it locally with:

    modal volume get steering-audit /audit ./data/steering_rl/audit
"""

import pathlib
import shlex
import socket
import subprocess
import time

import modal

# --- paths inside the container ---------------------------------------------
APP_DIR = "/app"  # repo root; PYTHONPATH roots hang off this
LIBERO_VENV = "/.venv"  # Python 3.8 — LIBERO client (mirrors libero.Dockerfile)
SERVER_VENV = "/server_venv"  # Python 3.11 — frozen JAX pi0.5 server
ASSETS_DIR = "/openpi_assets"  # persisted checkpoint cache (Volume)
AUDIT_DIR = "/audit"  # persisted audit log (Volume)

CUDA_IMAGE = "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04"
GPU = "A10G"  # 24GB, ~2-3x T4 for pi0.5 inference; image is GPU-agnostic so this is free to change
SERVER_PORT = 8000

# Repo-root-relative dirs/files baked into the image. Only meaningful LOCALLY (at
# image-definition time); Modal re-imports this module inside the container to run
# the function, where __file__ is /root/audit.py and these paths are unused — so
# fall back to a non-crashing value there instead of indexing a missing parent.
_HERE = pathlib.Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) >= 3 else _HERE.parent


def _p(*parts: str) -> str:
    return str(REPO.joinpath(*parts))


# --- image: both venvs in one container -------------------------------------
# Built with Modal's native builder (not from_dockerfile) so each venv gets its
# own inline env, and to avoid the libero Dockerfile's heredocs / COPY --from.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")  # add_python = Modal's own runtime
    .apt_install(
        # union of serve_policy.Dockerfile + libero.Dockerfile system deps
        "git", "git-lfs", "build-essential", "clang", "make", "g++", "curl",
        "libosmesa6-dev", "libgl1-mesa-glx", "libegl1", "libglew-dev",
        "libglfw3-dev", "libgles2-mesa-dev", "libglib2.0-0", "libsm6",
        "libxrender1", "libxext6",
    )
    .pip_install("uv==0.5.1")  # the same uv the Dockerfiles pin
    # ---- files needed to BUILD the two venvs (small; added first for caching) ----
    .add_local_file(_p("examples", "libero", "requirements.txt"), "/tmp/requirements.txt", copy=True)
    .add_local_file(_p("third_party", "libero", "requirements.txt"), "/tmp/requirements-libero.txt", copy=True)
    .add_local_file(_p("packages", "openpi-client", "pyproject.toml"), "/tmp/openpi-client/pyproject.toml", copy=True)
    .add_local_file(_p("uv.lock"), f"{APP_DIR}/uv.lock", copy=True)
    .add_local_file(_p("pyproject.toml"), f"{APP_DIR}/pyproject.toml", copy=True)
    .add_local_dir(_p("packages", "openpi-client"), f"{APP_DIR}/packages/openpi-client", copy=True)
    .add_local_dir(_p("src"), f"{APP_DIR}/src", copy=True)
    # ---- LIBERO client venv (Python 3.8), mirrors libero.Dockerfile ----
    .run_commands(
        "echo 'setuptools<70' > /tmp/build-constraints.txt",
        f"UV_LINK_MODE=copy uv venv --python 3.8 {LIBERO_VENV}",
        # torch cu113 wheels + unsafe-best-match, per the Dockerfile
        f"UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT={LIBERO_VENV} uv pip sync"
        " /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml"
        " --extra-index-url https://download.pytorch.org/whl/cu113"
        " --index-strategy=unsafe-best-match --build-constraint /tmp/build-constraints.txt",
    )
    # ---- frozen JAX server venv (Python 3.11), mirrors serve_policy.Dockerfile ----
    # NOTE: transformers_replace copy is intentionally skipped (CLAUDE.md §3 — we
    # serve JAX, never a PyTorch VLA, so the patch buys nothing and mutates cache).
    # openpi's pyproject references these; the editable build validates they exist.
    # Placed here (after the libero venv, before the server venv) so they don't
    # invalidate the cached libero-venv layer.
    .add_local_file(_p("LICENSE"), f"{APP_DIR}/LICENSE", copy=True)
    .add_local_file(_p("README.md"), f"{APP_DIR}/README.md", copy=True)
    # We bake the repo into the image (no runtime volume mount like Docker), so we
    # install the openpi project itself at build time (drop --no-install-project).
    # The server then launches directly from the venv, fully offline/deterministic.
    .run_commands(
        f"UV_LINK_MODE=copy uv venv --python 3.11.9 {SERVER_VENV}",
        f"cd {APP_DIR} && UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT={SERVER_VENV}"
        " GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev",
    )
    # ---- LIBERO runtime config + EGL vendor json (shipped as static files) ----
    .add_local_file(_p("steering_rl", "modal", "files", "libero_config.yaml"), "/tmp/libero/config.yaml", copy=True)
    .add_local_file(
        _p("steering_rl", "modal", "files", "egl_nvidia.json"),
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        copy=True,
    )
    # ---- runtime code (added last so edits don't rebuild the venvs) ----
    # (examples/ is intentionally not baked — the client reimplements the libero
    #  loop in steering_rl/libero/episode.py and imports nothing from examples/.)
    .add_local_dir(_p("scripts"), f"{APP_DIR}/scripts", copy=True)
    .add_local_dir(_p("third_party", "libero"), f"{APP_DIR}/third_party/libero", copy=True)
    .add_local_dir(_p("steering_rl"), f"{APP_DIR}/steering_rl", copy=True)
    .workdir(APP_DIR)
    .env(
        {
            # EGL/MuJoCo rendering for the client (harmless to the server).
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "LIBERO_CONFIG_PATH": "/tmp/libero",
        }
    )
)

app = modal.App("steering-rl-audit")
# Volumes: checkpoint cache (download pi0.5 once) + resumable audit log.
assets_vol = modal.Volume.from_name("openpi-assets", create_if_missing=True)
audit_vol = modal.Volume.from_name("steering-audit", create_if_missing=True)
# Publishes the frozen server's public tunnel address so a LOCAL client can reach it.
tunnel_info = modal.Dict.from_name("steering-tunnel", create_if_missing=True)

SERVE_GPU = "T4"  # inference-only (no rendering) -> cheapest GPU that fits pi0.5's >8GB floor


def _wait_for_port(host: str, port: int, timeout_s: float, server: "subprocess.Popen | None" = None) -> None:
    """Block until the server accepts TCP connections (or raise on timeout).

    If ``server`` is given, fail fast when the subprocess exits (crash on
    startup) instead of polling a dead port for the full timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"pi0.5 server exited with code {server.returncode} before opening :{port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(3.0)
    raise TimeoutError(f"pi0.5 server did not open {host}:{port} within {timeout_s:.0f}s")


@app.function(
    image=image,
    gpu=GPU,
    # Reserve ample system RAM: restoring the ~3.3B-param pi0.5 JAX checkpoint peaks
    # well above its on-disk size, and the default container RAM OOM-kills it (exit 137).
    memory=32768,
    # Exactly one container ever: two concurrent runs would both append to the same
    # volume JSONL and race/duplicate records. retries=0 so a failed call never spawns
    # a retry that races a manual re-spawn (which is how a phantom 2nd container arose).
    max_containers=1,
    retries=0,
    volumes={ASSETS_DIR: assets_vol, AUDIT_DIR: audit_vol},
    timeout=24 * 60 * 60,  # 24h (Modal max) so a multi-hour sweep finishes in one detached run; resumable anyway
)
def run_audit(client_args: str = "", server_wait_s: float = 1200.0, mujoco_gl: str = "osmesa") -> str:
    """Boot the frozen server, then run the audit sweep against it.

    ``client_args`` is forwarded verbatim to ``run_audit.py`` (tyro CLI), e.g.
    ``"--episodes 3"``. ``--host/--port/--out-dir`` are injected here so the
    client always targets the local server and writes to the audit Volume.
    ``mujoco_gl`` selects the client's MuJoCo render backend: ``osmesa`` (CPU,
    the robust headless default — EGL gives GL_FRAMEBUFFER_UNSUPPORTED on Modal,
    and glx needs an X display that a headless container lacks).
    """
    import os

    # ---- start the frozen JAX pi0.5 server (background) ----
    # openpi is installed into SERVER_VENV at build, so launch its python directly.
    server_env = {
        **os.environ,
        "OPENPI_DATA_HOME": ASSETS_DIR,  # persist the gs://openpi-assets download
        "IS_DOCKER": "true",
    }
    server_env.pop("MUJOCO_GL", None)  # server doesn't render
    server_env.pop("PYTHONPATH", None)  # use the installed package, not the libero paths
    server = subprocess.Popen(
        [f"{SERVER_VENV}/bin/python", "scripts/serve_policy.py", "--env", "LIBERO"],
        cwd=APP_DIR,
        env=server_env,
    )
    try:
        print(f"[modal] waiting up to {server_wait_s:.0f}s for pi0.5 server on :{SERVER_PORT} ...", flush=True)
        _wait_for_port("127.0.0.1", SERVER_PORT, server_wait_s, server=server)
        print("[modal] server is up; launching audit client", flush=True)

        # ---- run the audit client (foreground, Python 3.8 libero venv) ----
        client_env = {
            **os.environ,
            "PYTHONPATH": f"{APP_DIR}:{APP_DIR}/packages/openpi-client/src:{APP_DIR}/third_party/libero",
            "MUJOCO_GL": mujoco_gl,  # override the image's egl default (see docstring)
            "PYOPENGL_PLATFORM": mujoco_gl,
        }
        cmd = [
            f"{LIBERO_VENV}/bin/python", "steering_rl/libero/run_audit.py",
            "--host", "127.0.0.1", "--port", str(SERVER_PORT), "--out-dir", AUDIT_DIR,
            *shlex.split(client_args),
        ]
        result = subprocess.run(cmd, cwd=APP_DIR, env=client_env, check=False)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        audit_vol.commit()  # make sure the JSONL + summary are persisted

    if result.returncode != 0:
        raise RuntimeError(f"audit client exited with code {result.returncode}")
    summary_path = pathlib.Path(AUDIT_DIR) / "audit_summary.json"
    return summary_path.read_text() if summary_path.exists() else "(no summary written)"


def _start_frozen_server():
    """Launch the frozen JAX pi0.5 server subprocess and wait for it to listen."""
    import os

    server_env = {**os.environ, "OPENPI_DATA_HOME": ASSETS_DIR, "IS_DOCKER": "true"}
    server_env.pop("MUJOCO_GL", None)
    server_env.pop("PYTHONPATH", None)
    server = subprocess.Popen(
        [f"{SERVER_VENV}/bin/python", "scripts/serve_policy.py", "--env", "LIBERO"],
        cwd=APP_DIR,
        env=server_env,
    )
    _wait_for_port("127.0.0.1", SERVER_PORT, 1200.0, server=server)
    return server


@app.function(
    image=image,
    gpu=SERVE_GPU,
    memory=32768,
    max_containers=1,
    retries=0,
    volumes={ASSETS_DIR: assets_vol},
    timeout=24 * 60 * 60,
)
def serve(minutes: float = 180.0) -> None:
    """Serve the frozen pi0.5 over a public TCP tunnel for a LOCAL client.

    This is the frugal Phase C/D/E path: LIBERO rollouts + rendering + the RL loop
    run locally (free, on the 4070 via MUJOCO_GL=glx), and only pi0.5 inference
    runs here. Publishes the tunnel address to the ``steering-tunnel`` Dict; a
    local client reads it and connects with ``ws://host:port``. Serves for
    ``minutes`` (or until the app is stopped, which also stops billing).
    """
    server = _start_frozen_server()
    print(f"[serve] pi0.5 up; opening public tunnel to :{SERVER_PORT}", flush=True)
    try:
        with modal.forward(SERVER_PORT, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            tunnel_info["address"] = {"host": host, "port": port, "ts": time.time()}
            print(f"[serve] TUNNEL ws://{host}:{port}  (serving {minutes:.0f} min)", flush=True)
            deadline = time.time() + minutes * 60
            while time.time() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"pi0.5 server exited with code {server.returncode}")
                time.sleep(10)
    finally:
        tunnel_info.pop("address", None)
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


@app.function(image=image, timeout=900)  # no gpu= -> CPU only, ~free
def check_render(backend: str = "osmesa") -> str:
    """Cheap, GPU-free validation that MuJoCo/LIBERO can render with ``backend``.

    osmesa is pure-CPU software rendering, so this needs no GPU and no server —
    just make_env + reset + one predicate read in the 3.8 venv. Run before a
    (paid) GPU audit to confirm the render backend works:
        modal run steering_rl/modal/audit.py::check_render --backend osmesa
    """
    import os

    env = {
        **os.environ,
        "MUJOCO_GL": backend,
        "PYOPENGL_PLATFORM": backend,
        "PYTHONPATH": f"{APP_DIR}:{APP_DIR}/packages/openpi-client/src:{APP_DIR}/third_party/libero",
    }
    script = (
        "from steering_rl.libero import tasks, predicates;"
        "from steering_rl.libero.episode import make_env;"
        "e, _, s = make_env(tasks.TASKS[0].name, seed=tasks.ENV_SEED);"
        "e.reset(); e.set_init_state(s[0]);"
        "pe = predicates.find_problem_env(e);"
        "print('RENDER_OK', predicates.predicate_vector(pe).tolist())"
    )
    result = subprocess.run([f"{LIBERO_VENV}/bin/python", "-c", script], cwd=APP_DIR, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"render check failed for backend={backend} (exit {result.returncode})")
    return f"render OK with backend={backend}"


@app.local_entrypoint()
def main(episodes: int | None = None, tasks_limit: int | None = None, mujoco_gl: str = "osmesa", extra: str = "") -> None:
    """Launch the audit on Modal and print the gate summary.

    --episodes N     episodes per prompt (default: full 20 from tasks.NUM_EPISODES)
    --tasks-limit K  audit only the first K locked tasks (smoke tests; default all 5)
    --mujoco-gl B    client MuJoCo render backend (default osmesa; egl fails on Modal)
    --extra "..."    any extra flags passed straight to run_audit.py
    """
    args = []
    if episodes is not None:
        args += ["--episodes", str(episodes)]
    if tasks_limit is not None:
        args += ["--tasks-limit", str(tasks_limit)]
    if extra:
        args.append(extra)
    client_args = " ".join(args)
    print(f"[local] launching Modal audit with client_args={client_args!r} mujoco_gl={mujoco_gl}")
    summary = run_audit.remote(client_args, mujoco_gl=mujoco_gl)
    print("\n=== audit_summary.json ===")
    print(summary)
