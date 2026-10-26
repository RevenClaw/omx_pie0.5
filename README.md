# vla_layer

π0.5 (openpi) fine-tuning and serving for the OpenMANIPULATOR-X. Runs on the GPU server and needs no ROS.
The plan is in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Phase 0: server setup

```bash
git clone --recurse-submodules <this-repo-url> vla_layer && cd vla_layer
cp .env.example .env && nano .env          # set OPENPI_DATA_HOME to a disk with ≥150 GB free
bash scripts/setup_server.sh               # uv + openpi env + omx_pi05 (no sudo)
source scripts/env.sh                      # every new shell
bash scripts/gpu_check.sh train            # is the A30 free?
python scripts/download_checkpoints.py     # pi05_base + pi05_droid
python scripts/verify_env.py               # Phase 0 exit check
```
