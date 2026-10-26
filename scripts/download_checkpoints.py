"""Pre-fetch the pi0.5 checkpoints into $OPENPI_DATA_HOME so training and serving don't stall on first use.

    source scripts/env.sh
    python scripts/download_checkpoints.py            # pi05_base + pi05_droid
    python scripts/download_checkpoints.py --all      # also pi05_libero
    python scripts/download_checkpoints.py pi05_base  # only one

Downloads from the public gs://openpi-assets bucket (no Google account needed). Re-running is a no-op once cached.
"""

import argparse
import pathlib
import shutil
import sys
import time

from openpi.shared import download

CHECKPOINTS = {
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",  # the one we fine-tune (Phase 6)
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",  # zero-shot smoke test (Phase 1)
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",  # optional LIBERO sim check (Phase 1)
}
DEFAULT = ["pi05_base", "pi05_droid"]


def dir_size_gb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"checkpoints to fetch: {', '.join(CHECKPOINTS)}")
    parser.add_argument("--all", action="store_true", help="fetch every checkpoint in the list")
    args = parser.parse_args()
    names = list(CHECKPOINTS) if args.all else (args.names or DEFAULT)
    unknown = [n for n in names if n not in CHECKPOINTS]
    if unknown:
        parser.error(f"unknown checkpoint(s) {unknown}; choose from {list(CHECKPOINTS)}")

    cache = download.get_cache_dir()
    free_gb = shutil.disk_usage(cache).free / 1e9
    print(f"📁 Cache: {cache}  ({free_gb:.0f} GB free)")
    if free_gb < 50:
        print("⚠️  Less than 50 GB free here. Set OPENPI_DATA_HOME in .env to a bigger disk.")

    for name in names:
        url = CHECKPOINTS[name]
        print(f"📥 {name}: {url}")
        t0 = time.time()
        path = download.maybe_download(url)
        print(f"✅ {name} → {path}  ({dir_size_gb(path):.1f} GB, {time.time() - t0:.0f} s)")
        for sub in ("params", "assets"):
            if not (path / sub).exists():
                print(f"⚠️  {name} has no '{sub}/' folder; the checkpoint layout may have changed upstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
