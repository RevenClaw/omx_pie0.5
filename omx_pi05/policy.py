"""OM-X input/output transforms for openpi (modelled on openpi/policies/libero_policy.py).

Used for both training (after the dataset repack) and serving (directly on what the robot client sends):
    {"observation/front_image": uint8 (224,224,3), "observation/wrist_image": ..., "observation/state": (5,), "prompt": str}
There is no wrist camera for now (wrist_camera=False): the wrist slot is zeros and masked out, so the model ignores it, and
"observation/wrist_image" may be sent (it is ignored) or left out.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from omx_pi05 import constants as C


def make_omx_example() -> dict:
    """Random observation in the client format (for tests and smoke checks)."""
    return {
        C.OBS_FRONT: np.random.randint(256, size=(C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8),
        C.OBS_WRIST: np.random.randint(256, size=(C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8),
        C.OBS_STATE: np.array([*C.HOME, 0.0], dtype=np.float32),
        C.PROMPT: "pick up the cube",
    }


def _parse_image(image) -> np.ndarray:
    # LeRobot returns images as float32 (C,H,W) in [0,1]; the client sends uint8 (H,W,C). Normalize to uint8 HWC.
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OMXInputs(transforms.DataTransformFn):
    """front → base_0_rgb; right_wrist_0_rgb = zeros, masked out for pi0/pi0.5.

    wrist_camera=False (default, no wrist camera yet): left_wrist_0_rgb is also zeros and masked out, whatever the data holds.
    wrist_camera=True: wrist → left_wrist_0_rgb, unmasked. Training and serving must use the same setting.
    """

    model_type: _model.ModelType
    wrist_camera: bool = False

    def __call__(self, data: dict) -> dict:
        front = _parse_image(data[C.OBS_FRONT])
        wrist = _parse_image(data[C.OBS_WRIST]) if self.wrist_camera else np.zeros_like(front)
        # openpi masks padding images for pi0/pi0.5 but not pi0-FAST.
        padding_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": np.asarray(data[C.OBS_STATE], dtype=np.float32),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.wrist_camera else padding_mask,
                "right_wrist_0_rgb": padding_mask,
            },
        }
        if "actions" in data:  # training only
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if C.PROMPT in data:
            inputs["prompt"] = data[C.PROMPT]
        return inputs


@dataclasses.dataclass(frozen=True)
class OMXOutputs(transforms.DataTransformFn):
    """Inference only: drop the padding (model action_dim 32 → our 5)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : C.ACTION_DIM])}
