# Patched copy of openpi/examples/libero/Dockerfile for Phase B.
#
# openpi's Dockerfile is left untouched (CLAUDE.md rule: never edit files
# under src/openpi/, scripts/, or examples/ in place). This copy adds one
# fix: `easydict==1.9` fails to build under Python 3.8 because uv's default
# build isolation pulls the latest setuptools, and setuptools >=70 vendors a
# distutils that uses PEP 585 generics (`tuple[str, ...]`) requiring Python
# >=3.9. Pin the build-time setuptools via `--build-constraint` so the sdist
# build backend used for easydict stays Python-3.8-compatible.
#
# Build context must be the openpi repo root, e.g.:
#   docker build -f steering_rl/docker/libero.Dockerfile -t libero /path/to/openpi

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/libero/requirements.txt /tmp/requirements.txt
COPY ./third_party/libero/requirements.txt /tmp/requirements-libero.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Pin setuptools for sdist build isolation (see header comment).
RUN echo "setuptools<70" > /tmp/build-constraints.txt

# Install python dependencies.
RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match --build-constraint /tmp/build-constraints.txt
ENV PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero

# Create a default config file to avoid an input prompt from LIBERO's init script.
# https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/__init__.py
ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py $CLIENT_ARGS"]
