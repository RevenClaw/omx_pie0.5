"""Phase 2 (T2.9): will pi0.5 LoRA fit on this GPU, and at which batch size?

Runs openpi's real training loop with the real pi05_base weights and our planned LoRA model
(pi05=True, gemma_2b_lora + gemma_300m_lora, action_horizon=10) on openpi's FakeDataset (random tensors with
the exact model input shapes). No dataset or norm stats are needed. Nothing is saved.

    source scripts/env.sh
    bash scripts/gpu_check.sh train && python scripts/memory_test.py --batch-size 16

Run each batch size in its own process (JAX doesn't release memory cleanly after an out-of-memory error).
"""

import argparse
import importlib.util
import os
import sys
import time

import jax
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders

VLA_ROOT = os.environ.get("VLA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPENPI_DIR = os.path.join(VLA_ROOT, "third_party", "openpi")


def load_openpi_train_module():
    # openpi's train.py is a script, not a package module; load it by path so we can call main(config).
    spec = importlib.util.spec_from_file_location("openpi_train_script", os.path.join(OPENPI_DIR, "scripts", "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100, help="the first step includes JIT compilation (~minutes)")
    parser.add_argument("--action-horizon", type=int, default=10)
    args = parser.parse_args()

    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=args.action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    cfg = _config.TrainConfig(
        name="pi05_omx_lora_memtest",
        exp_name=f"bs{args.batch_size}",
        model=model,
        data=_config.FakeDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        log_interval=10,
        save_interval=10**9,
        checkpoint_base_dir=os.path.join(VLA_ROOT, "checkpoints", "memtest"),
        overwrite=True,
        wandb_enabled=False,
    )

    train = load_openpi_train_module()
    train._checkpoints.save_state = lambda *a, **k: None  # memory test only: skip writing a multi-GB checkpoint

    frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75 (JAX default)")
    print(f"🧪 pi0.5 LoRA memory test: batch={args.batch_size} steps={args.steps} horizon={args.action_horizon} mem_fraction={frac}")
    t0 = time.time()
    try:
        train.main(cfg)
    except Exception as e:  # noqa: BLE001 - report OOM (XlaRuntimeError RESOURCE_EXHAUSTED) cleanly
        oom = "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
        print(f"\n❌ batch={args.batch_size}: {'OUT OF MEMORY' if oom else type(e).__name__}: {str(e)[:300]}")
        return 1
    wall = time.time() - t0

    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 2**30
    limit = stats.get("bytes_limit", 0) / 2**30
    print(f"\n✅ batch={args.batch_size} fits. Peak {peak:.2f} GiB of {limit:.2f} GiB JAX limit ({100 * peak / limit:.0f}%).")
    print(f"   Wall time {wall / 60:.1f} min for {args.steps} steps (includes weight load + compile; read s/it from the tqdm bar).")
    print("   Log batch, peak memory and s/it in docs/experiments.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
