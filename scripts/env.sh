# Source this in every shell on the server:   source scripts/env.sh
# Sets paths, loads .env overrides and activates openpi's uv virtualenv.

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_ROOT
export OPENPI_DIR="$VLA_ROOT/third_party/openpi"

# Machine-specific overrides (not committed), e.g. OPENPI_DATA_HOME=/data/openpi
if [ -f "$VLA_ROOT/.env" ]; then
    set -a; . "$VLA_ROOT/.env"; set +a
fi

# Where openpi caches checkpoints (pi05_base is several GB). Put it on a big disk.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

# Port of the policy websocket server (8000, openpi's default, is taken on the A30 server).
export VLA_POLICY_PORT="${VLA_POLICY_PORT:-4000}"

# Let JAX use 90% of GPU memory instead of the default 75% (needed for LoRA on 24 GB).
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [ -f "$OPENPI_DIR/.venv/bin/activate" ]; then
    . "$OPENPI_DIR/.venv/bin/activate"
else
    echo "⚠️  openpi venv not found at $OPENPI_DIR/.venv. Run scripts/setup_server.sh first."
fi
