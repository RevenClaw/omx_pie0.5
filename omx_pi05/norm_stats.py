"""Normalization stats for an OM-X config, same computation as openpi's scripts/compute_norm_stats.py (torch-dataset path).

Lives in this importable package rather than calling openpi's script: openpi's data loader starts worker processes that
re-import every transform they receive, and a class defined inside a script loaded by file path (openpi's RemoveStrings)
cannot be re-imported there ("No module named 'openpi_scripts_compute_norm_stats'").
"""

import dataclasses

import numpy as np
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm

from omx_pi05.train_configs import get_config


@dataclasses.dataclass(frozen=True)
class RemoveStrings(transforms.DataTransformFn):
    """Drop string fields (the prompt): JAX can't batch them and norm stats don't need them."""

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute(config_name: str, max_frames: int | None = None) -> None:
    config = get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    # Same transforms as training up to (not including) normalization: repack, OMXInputs, DeltaActions.
    dataset = _data_loader.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()]
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches, shuffle = max_frames // config.batch_size, True
    else:
        num_batches, shuffle = len(dataset) // config.batch_size, False
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing stats"):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, {key: running.get_statistics() for key, running in stats.items()})
