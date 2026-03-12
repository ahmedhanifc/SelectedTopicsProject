from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_colormap
from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import get_cholecseg8k_colormap


@dataclass(frozen=True)
class DatasetConfig:
    dataset_type: str
    num_classes: int
    ignore_index: int | None
    background_index: int | None
    palette: np.ndarray


def _normalise_palette(palette: np.ndarray | list) -> np.ndarray:
    palette_arr = np.asarray(palette, dtype=np.uint8)
    if palette_arr.ndim == 1:
        palette_arr = palette_arr.reshape(-1, 3)
    return palette_arr


def get_dataset_config(dataset_type: str,
                       ignore_index: int | None = None,
                       background_index: int | None = None) -> DatasetConfig:
    dataset_key = dataset_type.upper()
    if dataset_key == "CADIS":
        default_ignore = 255
        default_background = None
        num_classes = 18
        palette = get_cadis_colormap()
    elif dataset_key == "CHOLECSEG8K":
        default_ignore = None
        default_background = 0
        num_classes = 13
        palette = get_cholecseg8k_colormap()
    elif dataset_key == "CATARACT1K":
        default_ignore = None
        default_background = 0
        num_classes = 14
        palette = get_cataract1k_colormap()
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return DatasetConfig(
        dataset_type=dataset_key,
        num_classes=num_classes,
        ignore_index=default_ignore if ignore_index is None else ignore_index,
        background_index=default_background if background_index is None else background_index,
        palette=_normalise_palette(palette),
    )
