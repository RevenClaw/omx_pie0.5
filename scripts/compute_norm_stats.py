"""Compute normalization stats (quantiles of state and delta-actions) for an OM-X config. CPU only, no GPU needed.

    source scripts/env.sh
    python scripts/compute_norm_stats.py pi05_omx_lora_debug
    python scripts/compute_norm_stats.py pi05_omx_lora_debug --max-frames 2000

Writes assets/<config>/<repo_id>/norm_stats.json. Re-run every time the dataset changes (plan T6.1).
"""

import tyro

from omx_pi05 import norm_stats


def main(config_name: tyro.conf.Positional[str], max_frames: int | None = None) -> None:
    norm_stats.compute(config_name, max_frames)


if __name__ == "__main__":
    tyro.cli(main)
