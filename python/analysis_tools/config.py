from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
    from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_colormap
    from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import get_cholecseg8k_colormap
except ModuleNotFoundError:
    def _fallback_colormap(num_classes: int, dataset_type: str | None = None) -> np.ndarray:
        if dataset_type == "CHOLECSEG8K":
            base_colors = np.array([
                [127, 127, 127],
                [255, 114, 114],
                [255, 160, 165],
                [186, 183, 75],
                [231, 70, 156],
                [210, 140, 140],
                [255, 255, 255],
                [255, 184, 138],
                [208, 168, 255],
                [129, 204, 184],
                [255, 214, 102],
                [145, 198, 255],
                [244, 143, 177],
            ], dtype=np.uint8)
        else:
            base_colors = np.array([
                [0, 0, 0],
                [230, 25, 75],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 190],
                [0, 128, 128],
                [230, 190, 255],
                [170, 110, 40],
                [255, 250, 200],
                [128, 0, 0],
                [170, 255, 195],
                [128, 128, 0],
                [255, 215, 180],
                [0, 0, 128],
            ], dtype=np.uint8)
        if num_classes <= len(base_colors):
            return base_colors[:num_classes]
        extra = []
        for idx in range(len(base_colors), num_classes):
            extra.append([(37 * idx) % 256, (97 * idx) % 256, (17 * idx) % 256])
        return np.vstack([base_colors, np.array(extra, dtype=np.uint8)])

    def get_cadis_colormap():
        return _fallback_colormap(18)

    def get_cholecseg8k_colormap():
        return _fallback_colormap(13, dataset_type="CHOLECSEG8K")

    def get_cataract1k_colormap():
        return _fallback_colormap(14)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_type: str
    num_classes: int
    ignore_index: int | None
    background_index: int | None
    palette: np.ndarray
    rgb_label_map: dict[tuple[int, int, int], int] | None = None


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
        rgb_label_map = {
            (255, 255, 255): 0,
            (50, 50, 50): 0,
            (11, 11, 11): 1,
            (21, 21, 21): 2,
            (13, 13, 13): 3,
            (12, 12, 12): 4,
            (31, 31, 31): 5,
            (23, 23, 23): 6,
            (24, 24, 24): 7,
            (25, 25, 25): 8,
            (32, 32, 32): 9,
            (22, 22, 22): 10,
            (33, 33, 33): 11,
            (5, 5, 5): 12,
        }
    elif dataset_key == "CATARACT1K":
        default_ignore = None
        default_background = 0
        num_classes = 14
        palette = get_cataract1k_colormap()
        rgb_label_map = None
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    if dataset_key == "CADIS":
        rgb_label_map = None

    return DatasetConfig(
        dataset_type=dataset_key,
        num_classes=num_classes,
        ignore_index=default_ignore if ignore_index is None else ignore_index,
        background_index=default_background if background_index is None else background_index,
        palette=_normalise_palette(palette),
        rgb_label_map=rgb_label_map,
    )
