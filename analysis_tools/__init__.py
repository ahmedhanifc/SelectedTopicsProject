"""Utilities for offline SASVi error analysis and optional inference exports."""

from .inference_export import compute_confidence_map, summarise_confidence_map, write_rows_to_csv

__all__ = [
    "DatasetConfig",
    "analyze_predictions",
    "compute_confidence_map",
    "get_dataset_config",
    "summarise_confidence_map",
    "write_rows_to_csv",
]


def __getattr__(name):
    if name in {"DatasetConfig", "get_dataset_config"}:
        from .config import DatasetConfig, get_dataset_config

        return {"DatasetConfig": DatasetConfig, "get_dataset_config": get_dataset_config}[name]
    if name == "analyze_predictions":
        from .error_analysis import analyze_predictions

        return analyze_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
