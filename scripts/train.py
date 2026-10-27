"""Train an OM-X config with openpi's own training loop and CLI (all openpi flags work, e.g. --batch-size, --resume).

    source scripts/env.sh
    bash scripts/gpu_check.sh train && python scripts/train.py pi05_omx_lora_debug --exp-name=dryrun --overwrite

Checkpoints go to checkpoints/<config>/<exp-name>/<step>/ (with the norm stats copied into <step>/assets/).
"""

import omx_pi05.train_configs  # noqa: F401  (registers the OM-X configs into openpi)
from omx_pi05 import openpi_scripts
from openpi.training import config as _config

if __name__ == "__main__":
    openpi_scripts.load("train").main(_config.cli())
