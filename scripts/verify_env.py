"""Phase 0 exit check: is this server ready for openpi pi0.5?

    source scripts/env.sh
    python scripts/verify_env.py

Prints one line per check and exits non-zero if any required check fails.
"""

import os
import pathlib
import shutil
import subprocess
import sys

REQUIRED_FREE_DISK_GB = 150
results: list[tuple[str, str, str]] = []  # (name, "ok" | "fail" | "warn", detail)


def check(name: str, required: bool = True):
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - report any failure as a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append((name, "ok" if ok else ("fail" if required else "warn"), detail))
        return fn

    return wrap


@check("python 3.11+ (openpi venv)")
def _():
    v = sys.version_info
    in_venv = "third_party/openpi/.venv" in sys.prefix
    return v >= (3, 11) and in_venv, f"{v.major}.{v.minor}.{v.micro} at {sys.prefix}"


@check("JAX sees the GPU")
def _():
    import jax

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    return bool(gpus), f"jax {jax.__version__}, devices={jax.devices()}"


@check("bf16 matmul on GPU")
def _():
    import jax
    import jax.numpy as jnp

    x = jnp.ones((4096, 4096), dtype=jnp.bfloat16)
    y = jax.jit(lambda a: a @ a)(x).block_until_ready()
    return float(y[0, 0]) == 4096.0, f"result[0,0]={float(y[0, 0])} on {y.devices()}"


@check("openpi imports + pi05 configs exist")
def _():
    from openpi.training import config

    names = [n for n in ("pi05_libero", "pi05_droid") if config.get_config(n)]
    return len(names) == 2, f"found {names}"


@check("openpi-client imports")
def _():
    from openpi_client import websocket_client_policy  # noqa: F401

    return True, "ok"


@check("lerobot (openpi's pinned version) imports")
def _():
    from lerobot.common.datasets import lerobot_dataset  # noqa: F401  (old module path used by openpi's pinned rev)

    return True, "lerobot.common.datasets.lerobot_dataset"


@check("omx_pi05 installed in this env")
def _():
    import omx_pi05

    return True, f"{omx_pi05.__version__} at {pathlib.Path(omx_pi05.__file__).parent}"


@check(f"disk ≥ {REQUIRED_FREE_DISK_GB} GB free at OPENPI_DATA_HOME")
def _():
    home = pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(home).free / 1e9
    return free >= REQUIRED_FREE_DISK_GB, f"{free:.0f} GB free at {home}"


@check("checkpoints cached (pi05_base, pi05_droid)")
def _():
    from openpi.shared import download

    cache = download.get_cache_dir() / "openpi-assets" / "checkpoints"
    missing = [n for n in ("pi05_base", "pi05_droid") if not (cache / n / "params").exists()]
    return not missing, f"missing {missing}; run scripts/download_checkpoints.py" if missing else f"in {cache}"


@check("ffmpeg (needed later for LeRobot video)", required=False)
def _():
    path = shutil.which("ffmpeg")
    return path is not None, path or "not on PATH"


@check("GPU free for training (gpu_check.sh train)", required=False)
def _():
    script = pathlib.Path(__file__).with_name("gpu_check.sh")
    r = subprocess.run(["bash", str(script), "train"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr.strip()


def main() -> int:
    width = max(len(n) for n, _, _ in results)
    icons = {"ok": "✅", "fail": "❌", "warn": "⚠️ "}
    for name, status, detail in results:
        print(f"{icons[status]} {name:<{width}}  {detail}")
    failed = [n for n, status, _ in results if status == "fail"]
    print("\n🎉 Phase 0 environment OK" if not failed else f"\n❌ {len(failed)} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
