#!/usr/bin/env bash
# One-time setup of the openpi environment on the GPU server. Safe to re-run.
# No sudo, no ROS, no Docker needed. uv downloads Python 3.11 itself.
#   bash scripts/setup_server.sh
set -euo pipefail

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$VLA_ROOT/third_party/openpi"

echo "📋 System"
( . /etc/os-release && echo "   OS: $PRETTY_NAME" )
echo "   Disk free at $HOME: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
command -v git >/dev/null || { echo "❌ git is required"; exit 1; }

# 1. uv (installs to ~/.local/bin, no root)
if ! command -v uv >/dev/null; then
    echo "📦 Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "   uv: $(uv --version)"

# 2. openpi submodule (+ its own aloha/libero submodules), at the commit pinned in this repo
echo "📥 Fetching openpi submodule"
git -C "$VLA_ROOT" submodule update --init --recursive
echo "   openpi @ $(git -C "$OPENPI_DIR" rev-parse --short HEAD)"

# 3. openpi env. GIT_LFS_SKIP_SMUDGE=1 is required for openpi's pinned lerobot dependency.
echo "🐍 Creating openpi env (first run downloads JAX/CUDA wheels, several GB)"
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. Our package into the same env. Re-run this script after every `uv sync`:
#    `uv sync` removes packages that are not in openpi's lockfile, including omx_pi05.
echo "🔗 Installing omx_pi05 into the openpi env"
uv pip install -e "$VLA_ROOT"

echo "✅ Setup done. Next:"
echo "   source scripts/env.sh"
echo "   python scripts/verify_env.py"
