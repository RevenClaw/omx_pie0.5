"""OM-X training configs, registered into openpi's config registry so openpi's own CLIs/tools find them by name.

    import omx_pi05.train_configs  # side effect: registers the configs below
    openpi.training.config.get_config("pi05_omx_lora_debug")

Model = the one validated by scripts/memory_test.py on the A30 (batch 16 → peak 17.7 / 21.2 GiB, 2026-10-27).
To add a real dataset: add one line to DATASETS. That creates `pi05_omx_lora_<name>`.
"""

import pathlib

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders

from omx_pi05 import constants as C
from omx_pi05.data_config import LeRobotOMXDataConfig

VLA_ROOT = pathlib.Path(__file__).resolve().parent.parent  # editable install → the repo checkout
ASSETS_DIR = VLA_ROOT / "assets"  # norm stats: assets/<config>/<repo_id>/norm_stats.json
CHECKPOINT_DIR = VLA_ROOT / "checkpoints"  # checkpoints/<config>/<exp_name>/<step>/

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
BATCH_SIZE = 16  # largest validated batch on the 24 GiB A30 (32 is estimated not to fit)

# LeRobot repo ids under $HF_LEROBOT_HOME. Real datasets get added here (e.g. "pick_cube": "local/omx_pick_cube").
DATASETS = {
    "synthetic": "local/omx_synthetic",
}


def lora_model() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=C.ACTION_HORIZON,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )


def lora_config(name: str, repo_id: str, **overrides) -> _config.TrainConfig:
    model = lora_model()
    kwargs = dict(
        name=name,
        model=model,
        data=LeRobotOMXDataConfig(
            repo_id=repo_id,
            base_config=_config.DataConfig(prompt_from_task=True, action_sequence_keys=(C.DS_ACTION,)),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE_PARAMS),
        # LoRA freeze filter: freezes both Gemma models except their LoRA weights. SigLIP and the action
        # projections stay fully trainable (see docs/phase2_report.md).
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,  # EMA would keep a second copy of the weights; no room on 24 GiB
        batch_size=BATCH_SIZE,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        log_interval=100,
        assets_base_dir=str(ASSETS_DIR),
        checkpoint_base_dir=str(CHECKPOINT_DIR),
        wandb_enabled=False,  # enable with --wandb-enabled after `wandb login`
    )
    kwargs.update(overrides)
    return _config.TrainConfig(**kwargs)


CONFIGS = [
    *(lora_config(f"pi05_omx_lora_{key}", repo_id) for key, repo_id in DATASETS.items()),
    # Pipeline dry run on the synthetic dataset: 200 steps, checkpoints at 100 and 199.
    lora_config(
        "pi05_omx_lora_debug",
        DATASETS["synthetic"],
        num_train_steps=200,
        save_interval=100,
        keep_period=None,
        log_interval=10,
    ),
]


def register() -> None:
    for cfg in CONFIGS:
        existing = _config._CONFIGS_DICT.get(cfg.name)  # noqa: SLF001 - openpi has no public registration API
        if existing is not None and existing is not cfg:
            raise ValueError(f"Config name {cfg.name!r} already exists in openpi")
        _config._CONFIGS_DICT[cfg.name] = cfg  # noqa: SLF001


def get_config(name: str) -> _config.TrainConfig:
    register()
    return _config.get_config(name)


register()
