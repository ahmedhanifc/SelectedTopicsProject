from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parent
APP_ROOT = PYTHON_ROOT.parent
SAM2_DIR = PYTHON_ROOT / "src" / "sam2"
EVAL_SCRIPT = SAM2_DIR / "eval_sasvi.py"


PRESETS = {
    "baseline_full": {
        "description": "Baseline run on the full dataset with disagreement disabled.",
        "mode": "baseline",
        "dataset_type": "CHOLECSEG8K",
    },
    "disagreement_no_bound_095": {
        "description": "Disagreement gate enabled without boundary distance, IoU threshold 0.95.",
        "mode": "disagreement",
        "dataset_type": "CHOLECSEG8K",
        "enable_disagreement_gate": True,
        "disagreement_iou_threshold": 0.95,
        "disagreement_min_label_area": 100,
        "disagreement_bad_frames": 2,
        "save_disagreement_visuals": True,
        "max_disagreement_visuals": 6,
    },
    "disagreement_no_bound_090_fg098": {
        "description": "Disagreement gate enabled without boundary distance, IoU threshold 0.90 and foreground IoU threshold 0.98.",
        "mode": "disagreement",
        "dataset_type": "CHOLECSEG8K",
        "enable_disagreement_gate": True,
        "disagreement_iou_threshold": 0.90,
        "disagreement_foreground_iou_threshold": 0.98,
        "disagreement_min_label_area": 100,
        "disagreement_bad_frames": 2,
        "save_disagreement_visuals": True,
        "max_disagreement_visuals": 6,
    },
    "disagreement_with_bound_090_fg098": {
        "description": "Disagreement gate enabled with boundary distance, IoU threshold 0.90 and foreground IoU threshold 0.98.",
        "mode": "disagreement-boundary",
        "dataset_type": "CHOLECSEG8K",
        "enable_disagreement_gate": True,
        "enable_boundary_distance_gate": True,
        "disagreement_iou_threshold": 0.90,
        "disagreement_foreground_iou_threshold": 0.98,
        "disagreement_min_label_area": 100,
        "disagreement_bad_frames": 2,
        "boundary_distance_threshold": 20.0,
        "save_disagreement_visuals": True,
        "max_disagreement_visuals": 6,
    },
}


def resolve_repo_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((APP_ROOT / path).resolve())


def resolve_sam2_cfg(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(SAM2_DIR).as_posix()
        except ValueError as exc:
            raise ValueError(
                "--sam2-cfg must point to a config inside src/sam2 so Hydra can resolve it."
            ) from exc
    return path.as_posix()


def default_overseer_checkpoint(dataset_type: str) -> str | None:
    defaults = {
        "CHOLECSEG8K": APP_ROOT / "checkpoints" / "cholecseg8k_maskrcnn_best_val_f1.pth",
    }
    checkpoint = defaults.get(dataset_type)
    return str(checkpoint) if checkpoint is not None else None


def mode_defaults(mode: str) -> dict[str, object]:
    if mode == "baseline":
        return {
            "enable_disagreement_gate": False,
            "enable_boundary_distance_gate": False,
        }
    if mode == "disagreement":
        return {
            "enable_disagreement_gate": True,
            "enable_boundary_distance_gate": False,
        }
    if mode == "disagreement-boundary":
        return {
            "enable_disagreement_gate": True,
            "enable_boundary_distance_gate": True,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def base_defaults() -> dict[str, object]:
    return {
        "mode": "baseline",
        "device": "cpu",
        "dataset_type": "CHOLECSEG8K",
        "sam2_cfg": "configs/sam2.1_hiera_l.yaml",
        "sam2_checkpoint": str((SAM2_DIR / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt").resolve()),
        "overseer_type": "MaskRCNN",
        "overseer_checkpoint": None,
        "nnunet_checkpoint": None,
        "base_video_dir": str((APP_ROOT / "dataset").resolve()),
        "output_mask_dir": None,
        "analysis_output_dir": None,
        "gt_root_dir": None,
        "video_name": None,
        "video_names": None,
        "frame_name": None,
        "start_frame": None,
        "end_frame": None,
        "score_thresh": 0.0,
        "apply_postprocessing": False,
        "save_binary_mask": False,
        "overseer_mask_dir": None,
        "nnunet_mask_dir": None,
        "enable_disagreement_gate": False,
        "disagreement_iou_threshold": 0.5,
        "disagreement_foreground_iou_threshold": 0.98,
        "disagreement_min_label_area": 100,
        "disagreement_bad_frames": 2,
        "enable_boundary_distance_gate": False,
        "boundary_distance_threshold": 20.0,
        "save_disagreement_visuals": False,
        "max_disagreement_visuals": 10,
    }


def build_run_name(config: dict[str, object], explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    mode = str(config["mode"]).replace("-", "_")
    dataset = str(config["dataset_type"]).lower()
    parts = [mode, dataset]

    if config.get("enable_disagreement_gate"):
        iou = int(round(float(config["disagreement_iou_threshold"]) * 100))
        fg_iou = int(round(float(config["disagreement_foreground_iou_threshold"]) * 100))
        parts.append(f"iou{iou:02d}")
        parts.append(f"fg{fg_iou:02d}")
        parts.append(f"mina{config['disagreement_min_label_area']}")
        parts.append(f"bad{config['disagreement_bad_frames']}")

    if config.get("enable_boundary_distance_gate"):
        parts.append("bound")

    if config.get("video_name"):
        parts.append(str(config["video_name"]))
    elif config.get("frame_name"):
        parts.append(str(config["frame_name"]))
    elif config.get("video_names"):
        parts.append("multi")
    else:
        parts.append("all")

    return "_".join(parts)


def apply_cli_overrides(config: dict[str, object], args: argparse.Namespace) -> None:
    scalar_overrides = {
        "mode": args.mode,
        "device": args.device,
        "dataset_type": args.dataset,
        "sam2_cfg": resolve_sam2_cfg(args.sam2_cfg) if args.sam2_cfg is not None else None,
        "sam2_checkpoint": resolve_repo_path(args.sam2_checkpoint) if args.sam2_checkpoint is not None else None,
        "overseer_type": args.overseer_type,
        "overseer_checkpoint": resolve_repo_path(args.overseer_checkpoint) if args.overseer_checkpoint is not None else None,
        "nnunet_checkpoint": resolve_repo_path(args.nnunet_checkpoint) if args.nnunet_checkpoint is not None else None,
        "base_video_dir": resolve_repo_path(args.base_video_dir) if args.base_video_dir is not None else None,
        "output_mask_dir": resolve_repo_path(args.output_mask_dir) if args.output_mask_dir is not None else None,
        "analysis_output_dir": resolve_repo_path(args.analysis_output_dir) if args.analysis_output_dir is not None else None,
        "gt_root_dir": resolve_repo_path(args.gt_root_dir) if args.gt_root_dir is not None else None,
        "video_name": args.video_name,
        "video_names": args.video_names,
        "frame_name": args.frame_name,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "score_thresh": args.score_thresh,
        "overseer_mask_dir": resolve_repo_path(args.overseer_mask_dir) if args.overseer_mask_dir is not None else None,
        "nnunet_mask_dir": resolve_repo_path(args.nnunet_mask_dir) if args.nnunet_mask_dir is not None else None,
        "disagreement_iou_threshold": args.disagreement_threshold,
        "disagreement_foreground_iou_threshold": args.disagreement_fg_threshold,
        "disagreement_min_label_area": args.disagreement_min_label_area,
        "disagreement_bad_frames": args.bad_frames,
        "boundary_distance_threshold": args.boundary_threshold,
        "max_disagreement_visuals": args.max_visuals,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            config[key] = value

    boolean_overrides = {
        "apply_postprocessing": args.apply_postprocessing,
        "save_binary_mask": args.save_binary_mask,
        "enable_disagreement_gate": args.enable_disagreement_gate,
        "enable_boundary_distance_gate": args.enable_boundary_distance_gate,
        "save_disagreement_visuals": args.save_visuals,
    }
    for key, value in boolean_overrides.items():
        if value is not None:
            config[key] = value


def finalize_config(args: argparse.Namespace) -> dict[str, object]:
    config = base_defaults()

    if args.preset is not None:
        config.update(PRESETS[args.preset])

    chosen_mode = args.mode if args.mode is not None else str(config["mode"])
    config.update(mode_defaults(chosen_mode))
    config["mode"] = chosen_mode

    apply_cli_overrides(config, args)

    if config.get("enable_boundary_distance_gate"):
        config["enable_disagreement_gate"] = True

    if config.get("video_name") is not None and config.get("video_names") is not None:
        raise ValueError("Use either --video-name or --video-names, not both.")

    dataset_type = str(config["dataset_type"])
    if args.overseer_checkpoint is None:
        default_checkpoint = default_overseer_checkpoint(dataset_type)
        config["overseer_checkpoint"] = default_checkpoint

    if config.get("overseer_checkpoint") is None:
        raise ValueError(
            f"No default overseer checkpoint is configured for dataset {dataset_type}. "
            "Pass --overseer-checkpoint explicitly."
        )

    run_name = build_run_name(config, args.run_name)
    if config.get("output_mask_dir") is None:
        config["output_mask_dir"] = str((APP_ROOT / f"output_masks_{run_name}").resolve())
    if config.get("analysis_output_dir") is None:
        config["analysis_output_dir"] = str((APP_ROOT / f"analysis_output_{run_name}").resolve())

    return config


def build_eval_command(config: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--device",
        str(config["device"]),
        "--sam2_cfg",
        str(config["sam2_cfg"]),
        "--sam2_checkpoint",
        str(config["sam2_checkpoint"]),
        "--overseer_checkpoint",
        str(config["overseer_checkpoint"]),
        "--overseer_type",
        str(config["overseer_type"]),
        "--dataset_type",
        str(config["dataset_type"]),
        "--base_video_dir",
        str(config["base_video_dir"]),
        "--output_mask_dir",
        str(config["output_mask_dir"]),
        "--analysis_output_dir",
        str(config["analysis_output_dir"]),
        "--score_thresh",
        str(config["score_thresh"]),
        "--disagreement_iou_threshold",
        str(config["disagreement_iou_threshold"]),
        "--disagreement_foreground_iou_threshold",
        str(config["disagreement_foreground_iou_threshold"]),
        "--disagreement_min_label_area",
        str(config["disagreement_min_label_area"]),
        "--disagreement_bad_frames",
        str(config["disagreement_bad_frames"]),
        "--boundary_distance_threshold",
        str(config["boundary_distance_threshold"]),
        "--max_disagreement_visuals",
        str(config["max_disagreement_visuals"]),
    ]

    optional_scalar_args = {
        "--nnunet_checkpoint": config.get("nnunet_checkpoint"),
        "--overseer_mask_dir": config.get("overseer_mask_dir"),
        "--nnunet_mask_dir": config.get("nnunet_mask_dir"),
        "--gt_root_dir": config.get("gt_root_dir"),
        "--video_name": config.get("video_name"),
        "--frame_name": config.get("frame_name"),
        "--start_frame": config.get("start_frame"),
        "--end_frame": config.get("end_frame"),
    }
    for flag, value in optional_scalar_args.items():
        if value is not None:
            command.extend([flag, str(value)])

    if config.get("video_names") is not None:
        command.append("--video_names")
        command.extend(str(name) for name in config["video_names"])

    optional_bool_flags = {
        "--apply_postprocessing": config.get("apply_postprocessing"),
        "--save_binary_mask": config.get("save_binary_mask"),
        "--enable_disagreement_gate": config.get("enable_disagreement_gate"),
        "--enable_boundary_distance_gate": config.get("enable_boundary_distance_gate"),
        "--save_disagreement_visuals": config.get("save_disagreement_visuals"),
    }
    for flag, enabled in optional_bool_flags.items():
        if enabled:
            command.append(flag)

    return command


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(args: argparse.Namespace) -> int:
    config = finalize_config(args)
    command = build_eval_command(config)

    print("Resolved run configuration:")
    print(f"  mode: {config['mode']}")
    print(f"  dataset: {config['dataset_type']}")
    print(f"  output_mask_dir: {config['output_mask_dir']}")
    print(f"  analysis_output_dir: {config['analysis_output_dir']}")
    print("\nCommand:")
    print(command_to_string(command))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=SAM2_DIR)
    return completed.returncode


def list_presets() -> int:
    print("Available presets:\n")
    for name, preset in PRESETS.items():
        print(f"- {name}: {preset['description']}")
    return 0


def add_run_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run", help="Run SASVi evaluation with root-level presets and overrides.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Optional preset to seed the run configuration.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "disagreement", "disagreement-boundary"],
        help="High-level run mode. If omitted, uses the preset mode or baseline.",
    )
    parser.add_argument("--run-name", help="Optional name used to derive default output directories.")
    parser.add_argument("--device", help="Execution device, e.g. cpu or cuda.")
    parser.add_argument(
        "--dataset",
        choices=["CADIS", "CHOLECSEG8K", "CATARACT1K"],
        help="Dataset type.",
    )
    parser.add_argument("--sam2-cfg", dest="sam2_cfg", help="SAM2 config path.")
    parser.add_argument("--sam2-checkpoint", help="SAM2 checkpoint path.")
    parser.add_argument(
        "--overseer-type",
        choices=["MaskRCNN", "DETR", "Mask2Former"],
        help="Overseer model type.",
    )
    parser.add_argument("--overseer-checkpoint", help="Overseer checkpoint path.")
    parser.add_argument("--nnunet-checkpoint", help="Optional nnUNet checkpoint path.")
    parser.add_argument("--base-video-dir", help="Root directory containing the video folders.")
    parser.add_argument("--output-mask-dir", help="Output directory for predicted masks.")
    parser.add_argument("--analysis-output-dir", help="Output directory for reports and metadata.")
    parser.add_argument("--gt-root-dir", help="Optional root directory containing ground-truth masks.")
    parser.add_argument("--video-name", help="Single video folder to run.")
    parser.add_argument("--video-names", nargs="+", help="Multiple video folders to run.")
    parser.add_argument("--frame-name", help="Single frame stem to run.")
    parser.add_argument("--start-frame", type=int, help="Optional first frame index.")
    parser.add_argument("--end-frame", type=int, help="Optional last frame index.")
    parser.add_argument("--score-thresh", type=float, help="Output mask logits threshold.")
    parser.add_argument("--overseer-mask-dir", help="Optional output dir for overseer masks.")
    parser.add_argument("--nnunet-mask-dir", help="Optional output dir for nnUNet masks.")
    parser.add_argument(
        "--apply-postprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable postprocessing.",
    )
    parser.add_argument(
        "--save-binary-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable per-object binary mask export.",
    )
    parser.add_argument(
        "--enable-disagreement-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable disagreement-gated re-prompting.",
    )
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        help="SAM2 vs Overseer disagreement IoU threshold.",
    )
    parser.add_argument(
        "--disagreement-fg-threshold",
        type=float,
        help="SAM2 vs Overseer foreground IoU threshold required for a bad disagreement frame.",
    )
    parser.add_argument(
        "--disagreement-min-label-area",
        type=int,
        help="Ignore labels smaller than this many pixels when computing disagreement IoU.",
    )
    parser.add_argument("--bad-frames", type=int, help="Consecutive bad frames required before re-prompting.")
    parser.add_argument(
        "--enable-boundary-distance-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable boundary-distance gating.",
    )
    parser.add_argument("--boundary-threshold", type=float, help="Boundary-distance threshold in pixels.")
    parser.add_argument(
        "--save-visuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable disagreement visuals.",
    )
    parser.add_argument("--max-visuals", type=int, help="Maximum disagreement visuals per video.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    parser.set_defaults(func=run_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Root entrypoint for SASVi experiment runs.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    add_run_subcommand(subparsers)

    list_parser = subparsers.add_parser("list-presets", help="List the available run presets.")
    list_parser.set_defaults(func=lambda _args: list_presets())
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
