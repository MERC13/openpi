"""Serve pi05_base (zero-shot) on LIBERO — the Phase-C §9 fallback experiment.

pi05_libero IS pi05_base fine-tuned on LIBERO (its config's weight_loader points at
pi05_base/params). The first audit found pi05_libero is strong but largely
prompt-robust — little headroom for learned steering. §9's fallback is to test a
checkpoint with better language-following. pi05_base is the same architecture but
NOT LIBERO-fine-tuned, so it should (a) follow language more and (b) be less
competent at LIBERO — a lower baseline = more headroom, IF it does the tasks at all.

We serve pi05_base's params with LIBERO's transforms + norm stats (exactly what the
pi05_libero server uses), via create_trained_policy's explicit norm_stats override.
This wraps openpi's public API only — no openpi files are edited (CLAUDE.md §4), and
nothing is trained or backpropped (CLAUDE.md §2).

NOTE: this is an experiment. pi05_base never saw LIBERO's embodiment/objects, and its
state normalization differs, so zero-shot competence is unknown — the audit measures it.

Run (in the server venv, from the repo root):
    python steering_rl/serving/serve_base_libero.py --port 8000
"""

import logging
import pathlib

import tyro

import openpi.policies.policy_config as _policy_config
import openpi.serving.websocket_policy_server as _websocket_policy_server
import openpi.shared.download as download
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config

# pi05_base provides params/; pi05_libero provides the LIBERO assets/ (norm stats).
BASE_PARAMS_CKPT = "gs://openpi-assets/checkpoints/pi05_base"
LIBERO_ASSETS_CKPT = "gs://openpi-assets/checkpoints/pi05_libero"


def main(port: int = 8000) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    # LIBERO transforms + data config (same as the pi05_libero server).
    config = _config.get_config("pi05_libero")
    data_config = config.data.create(config.assets_dirs, config.model)

    # LIBERO norm stats, loaded from the pi05_libero checkpoint's assets.
    libero_dir = pathlib.Path(download.maybe_download(LIBERO_ASSETS_CKPT))
    norm_stats = _checkpoints.load_norm_stats(libero_dir / "assets", data_config.asset_id)
    logging.info("Loaded LIBERO norm stats (asset_id=%s): %s", data_config.asset_id, norm_stats is not None)

    # pi05_base params + LIBERO norm stats + LIBERO transforms.
    policy = _policy_config.create_trained_policy(config, BASE_PARAMS_CKPT, norm_stats=norm_stats)

    server = _websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=port, metadata=policy.metadata
    )
    logging.info("Serving pi05_base-on-LIBERO on :%d", port)
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
