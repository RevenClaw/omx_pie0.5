"""How openpi reads an OM-X LeRobot dataset (modelled on openpi LeRobotLiberoDataConfig).

Pass `base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",))` (see train_configs.py) so the
prompt comes from the LeRobot task and the action chunk is read from the `action` feature.
"""

import dataclasses
import pathlib

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from typing_extensions import override

from omx_pi05 import constants as C
from omx_pi05.policy import OMXInputs, OMXOutputs

# Joints 1–4 are trained as deltas from the current state, the gripper stays absolute (plan D4).
DELTA_ACTION_MASK = _transforms.make_bool_mask(4, -1)  # (True, True, True, True, False)


@dataclasses.dataclass(frozen=True)
class LeRobotOMXDataConfig(_config.DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Training only: dataset feature names → the keys the robot client sends at inference.
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        C.OBS_FRONT: C.DS_FRONT,
                        C.OBS_WRIST: C.DS_WRIST,
                        C.OBS_STATE: C.DS_STATE,
                        "actions": C.DS_ACTION,  # with delta_timestamps the loader returns `action` as an (H, 5) chunk
                        C.PROMPT: C.PROMPT,  # added by PromptFromLeRobotTask (prompt_from_task=True)
                    }
                )
            ]
        )
        # Training *and* serving. push() prepends outputs, so serving runs AbsoluteActions, then OMXOutputs.
        data_transforms = _transforms.Group(
            inputs=[OMXInputs(model_type=model_config.model_type)],
            outputs=[OMXOutputs()],
        ).push(
            inputs=[_transforms.DeltaActions(DELTA_ACTION_MASK)],
            outputs=[_transforms.AbsoluteActions(DELTA_ACTION_MASK)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )
