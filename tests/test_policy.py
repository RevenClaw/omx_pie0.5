"""OM-X transforms inside openpi's real pipeline (needs the openpi env)."""

import numpy as np
import pytest

pytest.importorskip("openpi")

from openpi.models import model as _model  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from omx_pi05 import constants as C  # noqa: E402
from omx_pi05.policy import OMXInputs, OMXOutputs, make_omx_example  # noqa: E402
from omx_pi05.train_configs import get_config  # noqa: E402

H = C.ACTION_HORIZON


def dataset_sample():
    """What the LeRobot loader yields for one frame: float CHW images, state (5,), action chunk (H, 5), prompt."""
    rng = np.random.default_rng(0)
    state = np.array([*C.HOME, 0.3], dtype=np.float32)
    actions = (state + rng.normal(0, 0.05, (H, 5))).astype(np.float32)
    return {
        C.DS_FRONT: rng.random((3, C.IMAGE_SIZE, C.IMAGE_SIZE), dtype=np.float32),
        C.DS_WRIST: rng.random((3, C.IMAGE_SIZE, C.IMAGE_SIZE), dtype=np.float32),
        C.DS_STATE: state,
        C.DS_ACTION: actions,
        C.PROMPT: "pick up the cube",
    }


def test_inputs_shapes_and_masks():
    out = OMXInputs(model_type=_model.ModelType.PI05)(make_omx_example())
    assert out["state"].shape == (5,)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(im.shape == (224, 224, 3) and im.dtype == np.uint8 for im in out["image"].values())
    assert not out["image"]["right_wrist_0_rgb"].any()
    assert out["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    assert "actions" not in out and out["prompt"] == "pick up the cube"


def test_inputs_parse_lerobot_images():
    sample = make_omx_example()
    sample[C.OBS_FRONT] = np.ones((3, 224, 224), dtype=np.float32)  # LeRobot: float CHW in [0, 1]
    out = OMXInputs(model_type=_model.ModelType.PI05)(sample)
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3) and out["image"]["base_0_rgb"].max() == 255


def test_outputs_drop_padding():
    assert OMXOutputs()({"actions": np.zeros((H, 32))})["actions"].shape == (H, 5)


def test_training_pipeline_makes_joint_deltas_and_keeps_gripper_absolute():
    cfg = get_config("pi05_omx_lora_debug")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_cfg.prompt_from_task and tuple(data_cfg.action_sequence_keys) == (C.DS_ACTION,)
    assert data_cfg.use_quantile_norm  # pi0.5 uses quantile normalization
    sample = dataset_sample()
    raw_actions, state = sample[C.DS_ACTION].copy(), sample[C.DS_STATE].copy()
    x = sample
    for t in [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs]:
        x = t(x)
    np.testing.assert_allclose(x["actions"][:, :4], raw_actions[:, :4] - state[:4], atol=1e-6)
    np.testing.assert_allclose(x["actions"][:, 4], raw_actions[:, 4])
    assert x["prompt"] == "pick up the cube"


def test_serving_outputs_round_trip_to_absolute():
    cfg = get_config("pi05_omx_lora_debug")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    state = np.array([*C.HOME, 0.0], dtype=np.float32)
    deltas = np.zeros((H, 32), dtype=np.float32)
    deltas[:, :4], deltas[:, 4] = 0.01, 1.0
    x = {"state": np.pad(state, (0, 27)), "actions": deltas}  # model outputs are padded to 32
    for t in data_cfg.data_transforms.outputs:
        x = t(x)
    assert x["actions"].shape == (H, 5)
    expected_joints = np.tile(np.array(C.HOME) + 0.01, (H, 1))  # every step: current pose + delta
    np.testing.assert_allclose(x["actions"][:, :4], expected_joints, atol=1e-6)
    np.testing.assert_allclose(x["actions"][:, 4], np.ones(H))


def test_model_transforms_tokenize_and_pad():
    cfg = get_config("pi05_omx_lora_debug")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    x = dataset_sample()
    for t in [*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs, *data_cfg.model_transforms.inputs]:
        x = t(x)
    assert x["state"].shape == (32,) and x["actions"].shape == (H, 32)
    assert x["tokenized_prompt"].shape == (cfg.model.max_token_len,)
    assert isinstance(cfg.data, _config.DataConfigFactory)
