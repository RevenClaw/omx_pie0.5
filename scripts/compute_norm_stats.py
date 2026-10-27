"""Compute normalization stats (quantiles of state and delta-actions) for an OM-X config. CPU only, no GPU needed.

    source scripts/env.sh
    python scripts/compute_norm_stats.py pi05_omx_lora_debug
    python scripts/compute_norm_stats.py pi05_omx_lora_debug --max-frames 2000

Writes assets/<config>/<repo_id>/norm_stats.json. Re-run every time the dataset changes (plan T6.1).
"""

import tyro

import omx_pi05.train_configs  # noqa: F401  (registers the OM-X configs into openpi)
from omx_pi05 import openpi_scripts


def main(config_name: str, max_frames: int | None = None) -> None:
    openpi_scripts.load("compute_norm_stats").main(config_name, max_frames)


if __name__ == "__main__":
    tyro.cli(main)
