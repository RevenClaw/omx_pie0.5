import pytest

pytest.importorskip("openpi")

from openpi.training import config as _config  # noqa: E402

from omx_pi05 import constants as C  # noqa: E402
from omx_pi05 import train_configs  # noqa: E402


@pytest.mark.parametrize("name", ["pi05_omx_lora_synthetic", "pi05_omx_lora_debug"])
def test_registered_and_matches_memory_test(name):
    cfg = _config.get_config(name)  # registered into openpi's own registry
    m = cfg.model
    assert m.pi05 and m.action_horizon == C.ACTION_HORIZON and m.action_dim == 32
    assert m.paligemma_variant == "gemma_2b_lora" and m.action_expert_variant == "gemma_300m_lora"
    assert cfg.batch_size == 16 and cfg.ema_decay is None
    assert cfg.weight_loader.params_path == train_configs.PI05_BASE_PARAMS
    assert str(cfg.assets_dirs).startswith(str(train_configs.ASSETS_DIR))
    assert str(cfg.checkpoint_base_dir) == str(train_configs.CHECKPOINT_DIR)


def test_debug_is_short():
    cfg = _config.get_config("pi05_omx_lora_debug")
    assert cfg.num_train_steps == 200 and cfg.save_interval == 100
    assert cfg.data.repo_id == train_configs.DATASETS["synthetic"]
