"""Utilities for offline SASVi error analysis and optional inference exports."""

from .config import DatasetConfig, get_dataset_config
from .error_analysis import analyze_predictions
from .inference_export import compute_confidence_map, summarise_confidence_map, write_rows_to_csv

__all__ = [
    "DatasetConfig",
    "analyze_predictions",
    "compute_confidence_map",
    "get_dataset_config",
    "summarise_confidence_map",
    "write_rows_to_csv",
]
