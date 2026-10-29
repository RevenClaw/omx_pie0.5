"""Serve a trained OM-X checkpoint with openpi's websocket server (same CLI as openpi scripts/serve_policy.py).

    source scripts/env.sh
    bash scripts/gpu_check.sh serve && python scripts/serve.py policy:checkpoint \
        --policy.config=pi05_omx_lora_debug --policy.dir=checkpoints/pi05_omx_lora_debug/dryrun/199

The port defaults to $VLA_POLICY_PORT (4000) unless --port is given.
"""

import logging
import os
import sys

import tyro

import omx_pi05.train_configs  # noqa: F401  (registers the OM-X configs into openpi)
from omx_pi05 import openpi_scripts

if __name__ == "__main__":
    if not any(a == "--port" or a.startswith("--port=") for a in sys.argv[1:]):
        # Top-level option: must come before the `policy:checkpoint` subcommand, or tyro rejects it.
        sys.argv.insert(1, f"--port={os.environ.get('VLA_POLICY_PORT', '4000')}")
    serve_policy = openpi_scripts.load("serve_policy")
    logging.basicConfig(level=logging.INFO, force=True)
    serve_policy.main(tyro.cli(serve_policy.Args))
