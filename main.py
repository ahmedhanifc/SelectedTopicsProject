from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SAM2_DIR = REPO_ROOT / "src" / "sam2"
EVAL_SCRIPT = SAM2_DIR / "eval_sasvi.py"
PARALLEL_DISAGREEMENT_SCRIPT = REPO_ROOT / "parallelization_and_streaming" / "parallel_disagreement_sasvi.py"
TIMED_ORIGINAL_SCRIPT = REPO_ROOT / "parallelization_and_streaming" / "time_original_disagreement_sasvi.py"
ERROR_ANALYSIS_SCRIPT = REPO_ROOT / "analysis_tools" / "run_error_analysis.py"


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
    "parallel_disagreement_090": {
        "description": "Parallel disagreement pipeline with IoU threshold 0.90.",
        "mode": "parallel-disagreement",
        "dataset_type": "CHOLECSEG8K",
        "enable_disagreement_gate": True,
        "disagreement_iou_threshold": 0.90,
        "disagreement_bad_frames": 2,
    },
    "original_timed_disagreement_090": {
        "description": "Original disagreement pipeline with timing wrapper and IoU threshold 0.90.",
        "mode": "original-timed-disagreement",
        "dataset_type": "CHOLECSEG8K",
        "enable_disagreement_gate": True,
        "disagreement_iou_threshold": 0.90,
        "disagreement_bad_frames": 2,
    },
}


def resolve_repo_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


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
        "CHOLECSEG8K": REPO_ROOT / "checkpoints" / "cholecseg8k_maskrcnn_best_val_f1.pth",
    }
    checkpoint = defaults.get(dataset_type)
    return str(checkpoint) if checkpoint is not None else None


def default_sam2_checkpoint() -> str:
    checkpoint_path = SAM2_DIR / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    return str(checkpoint_path.resolve())


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
    if mode == "parallel-disagreement":
        return {
            "enable_disagreement_gate": True,
            "enable_boundary_distance_gate": False,
        }
    if mode == "original-timed-disagreement":
        return {
            "enable_disagreement_gate": True,
            "enable_boundary_distance_gate": False,
        }
    raise ValueError(f"Unsupported mode: {mode}")


def base_defaults() -> dict[str, object]:
    return {
        "mode": "baseline",
        "device": "cpu",
        "dataset_type": "CHOLECSEG8K",
        "sam2_cfg": "configs/sam2.1_hiera_l.yaml",
        "sam2_checkpoint": default_sam2_checkpoint(),
        "overseer_type": "MaskRCNN",
        "overseer_checkpoint": None,
        "nnunet_checkpoint": None,
        "base_video_dir": str((REPO_ROOT / "dataset").resolve()),
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
        "output_root": None,
        "analysis_output_root": None,
        "save_overseer_masks": False,
        "max_videos": None,
        "overseer_batch_size": 8,
        "save_queue_size": 32,
        "disable_tqdm": False,
    }


def is_parallel_mode(mode: str) -> bool:
    return mode in {"parallel-disagreement", "original-timed-disagreement"}


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
        "output_root": resolve_repo_path(args.output_root) if args.output_root is not None else None,
        "analysis_output_root": resolve_repo_path(args.analysis_output_root) if args.analysis_output_root is not None else None,
        "disagreement_iou_threshold": args.disagreement_threshold,
        "disagreement_foreground_iou_threshold": args.disagreement_fg_threshold,
        "disagreement_min_label_area": args.disagreement_min_label_area,
        "disagreement_bad_frames": args.bad_frames,
        "boundary_distance_threshold": args.boundary_threshold,
        "max_disagreement_visuals": args.max_visuals,
        "max_videos": args.max_videos,
        "overseer_batch_size": args.overseer_batch_size,
        "save_queue_size": args.save_queue_size,
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
        "save_overseer_masks": args.save_overseer_masks,
        "disable_tqdm": args.disable_tqdm,
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
    if is_parallel_mode(str(config["mode"])):
        if config.get("output_root") is None:
            default_output_root = (
                f"parallel_disagreement_frame_outputs_{run_name}"
                if config["mode"] == "parallel-disagreement"
                else f"disagreement_frame_outputs_{run_name}"
            )
            config["output_root"] = str((REPO_ROOT / default_output_root).resolve())
        if config.get("analysis_output_root") is None:
            default_analysis_root = (
                f"analysis_output_disagreement_parallel_{run_name}"
                if config["mode"] == "parallel-disagreement"
                else f"analysis_output_disagreement_original_{run_name}"
            )
            config["analysis_output_root"] = str((REPO_ROOT / default_analysis_root).resolve())
    else:
        if config.get("output_mask_dir") is None:
            config["output_mask_dir"] = str((REPO_ROOT / f"output_masks_{run_name}").resolve())
        if config.get("analysis_output_dir") is None:
            config["analysis_output_dir"] = str((REPO_ROOT / f"analysis_output_{run_name}").resolve())

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


def build_parallel_command(config: dict[str, object]) -> list[str]:
    mode = str(config["mode"])
    script_path = (
        PARALLEL_DISAGREEMENT_SCRIPT
        if mode == "parallel-disagreement"
        else TIMED_ORIGINAL_SCRIPT
    )
    command = [
        sys.executable,
        str(script_path),
        "--device",
        str(config["device"]),
        "--sam2_cfg",
        str((SAM2_DIR / str(config["sam2_cfg"])).resolve()),
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
        "--output_root",
        str(config["output_root"]),
        "--analysis_output_root",
        str(config["analysis_output_root"]),
        "--score_thresh",
        str(config["score_thresh"]),
        "--disagreement_iou_threshold",
        str(config["disagreement_iou_threshold"]),
        "--disagreement_bad_frames",
        str(config["disagreement_bad_frames"]),
        "--boundary_distance_threshold",
        str(config["boundary_distance_threshold"]),
        "--max_disagreement_visuals",
        str(config["max_disagreement_visuals"]),
    ]

    optional_scalar_args = {
        "--gt_root_dir": config.get("gt_root_dir"),
        "--video_name": config.get("video_name"),
        "--frame_name": config.get("frame_name"),
        "--start_frame": config.get("start_frame"),
        "--end_frame": config.get("end_frame"),
        "--max_videos": config.get("max_videos"),
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
        "--save_overseer_masks": config.get("save_overseer_masks"),
        "--enable_disagreement_gate": config.get("enable_disagreement_gate"),
        "--enable_boundary_distance_gate": config.get("enable_boundary_distance_gate"),
        "--save_disagreement_visuals": config.get("save_disagreement_visuals"),
    }
    for flag, enabled in optional_bool_flags.items():
        if enabled:
            command.append(flag)

    if mode == "parallel-disagreement":
        command.extend([
            "--overseer_batch_size",
            str(config["overseer_batch_size"]),
            "--save_queue_size",
            str(config["save_queue_size"]),
        ])
        if config.get("disable_tqdm"):
            command.append("--disable_tqdm")

    return command


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_error_analysis_output_root(run_name: str | None, pred_root: str) -> str:
    suffix = run_name or Path(pred_root).name
    return str((REPO_ROOT / "reports" / f"error_analysis_{suffix}").resolve())


def build_error_analysis_command(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    if args.confidence_low_threshold > args.confidence_medium_threshold:
        raise ValueError("--confidence-low-threshold must be <= --confidence-medium-threshold.")

    dataset = args.dataset or "CHOLECSEG8K"
    frames_root = resolve_repo_path(args.frames_root)
    pred_root = resolve_repo_path(args.pred_root)
    gt_root = resolve_repo_path(args.gt_root)
    confidence_root = resolve_repo_path(args.confidence_root) if args.confidence_root is not None else None
    output_root = (
        resolve_repo_path(args.output_root)
        if args.output_root is not None
        else build_error_analysis_output_root(args.run_name, pred_root)
    )

    command = [
        sys.executable,
        str(ERROR_ANALYSIS_SCRIPT),
        "--frames_root",
        str(frames_root),
        "--pred_root",
        str(pred_root),
        "--gt_root",
        str(gt_root),
        "--output_root",
        str(output_root),
        "--dataset_type",
        str(dataset),
        "--pred_mask_suffix",
        str(args.pred_mask_suffix),
        "--gt_mask_suffix",
        str(args.gt_mask_suffix),
        "--image_suffix",
        str(args.image_suffix),
        "--confidence_low_threshold",
        str(args.confidence_low_threshold),
        "--confidence_medium_threshold",
        str(args.confidence_medium_threshold),
    ]

    optional_scalar_args = {
        "--confidence_root": confidence_root,
        "--ignore_index": args.ignore_index,
        "--background_index": args.background_index,
    }
    for flag, value in optional_scalar_args.items():
        if value is not None:
            command.extend([flag, str(value)])

    resolved = {
        "dataset": dataset,
        "frames_root": frames_root,
        "pred_root": pred_root,
        "gt_root": gt_root,
        "confidence_root": confidence_root,
        "output_root": output_root,
    }
    return command, resolved


def run_command(args: argparse.Namespace) -> int:
    config = finalize_config(args)
    command = build_parallel_command(config) if is_parallel_mode(str(config["mode"])) else build_eval_command(config)

    print("Resolved run configuration:")
    print(f"  mode: {config['mode']}")
    print(f"  dataset: {config['dataset_type']}")
    if is_parallel_mode(str(config["mode"])):
        print(f"  output_root: {config['output_root']}")
        print(f"  analysis_output_root: {config['analysis_output_root']}")
    else:
        print(f"  output_mask_dir: {config['output_mask_dir']}")
        print(f"  analysis_output_dir: {config['analysis_output_dir']}")
    print("\nCommand:")
    print(command_to_string(command))

    if args.dry_run:
        return 0

    run_cwd = REPO_ROOT if is_parallel_mode(str(config["mode"])) else SAM2_DIR
    completed = subprocess.run(command, cwd=run_cwd)
    return completed.returncode


def run_error_analysis_command(args: argparse.Namespace) -> int:
    command, resolved = build_error_analysis_command(args)

    print("Resolved error-analysis configuration:")
    print(f"  dataset: {resolved['dataset']}")
    print(f"  frames_root: {resolved['frames_root']}")
    print(f"  pred_root: {resolved['pred_root']}")
    print(f"  gt_root: {resolved['gt_root']}")
    print(f"  output_root: {resolved['output_root']}")
    if resolved["confidence_root"] is not None:
        print(f"  confidence_root: {resolved['confidence_root']}")
    print("\nCommand:")
    print(command_to_string(command))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT)
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
        choices=["baseline", "disagreement", "disagreement-boundary", "parallel-disagreement", "original-timed-disagreement"],
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
    parser.add_argument("--output-root", help="Root output directory used by the parallel and timed-original runners.")
    parser.add_argument("--analysis-output-root", help="Root analysis directory used by the parallel and timed-original runners.")
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
    parser.add_argument("--max-videos", type=int, help="Optional cap on number of videos for parallel/timed-original runs.")
    parser.add_argument(
        "--save-visuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable disagreement visuals.",
    )
    parser.add_argument("--max-visuals", type=int, help="Maximum disagreement visuals per video.")
    parser.add_argument(
        "--save-overseer-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Overseer mask export for parallel/timed-original runs.",
    )
    parser.add_argument("--overseer-batch-size", type=int, help="Parallel Overseer precomputation batch size.")
    parser.add_argument("--save-queue-size", type=int, help="Parallel save queue size.")
    parser.add_argument(
        "--disable-tqdm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tqdm progress bars for the parallel runner.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    parser.set_defaults(func=run_command)


def add_error_analysis_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("error-analysis", help="Run offline error analysis and write reports under reports/ by default.")
    parser.add_argument(
        "--dataset",
        choices=["CADIS", "CHOLECSEG8K", "CATARACT1K"],
        default="CHOLECSEG8K",
        help="Dataset type used to decode RGB masks.",
    )
    parser.add_argument("--frames-root", required=True, help="Root directory containing frame folders.")
    parser.add_argument("--pred-root", required=True, help="Root directory containing predicted RGB masks.")
    parser.add_argument("--gt-root", required=True, help="Root directory containing ground-truth RGB masks.")
    parser.add_argument("--confidence-root", help="Optional root directory containing exported confidence maps.")
    parser.add_argument("--output-root", help="Optional directory for the report bundle. Defaults to reports/error_analysis_<run-name>.")
    parser.add_argument("--run-name", help="Optional name used to derive the default reports directory.")
    parser.add_argument("--pred-mask-suffix", default="_rgb_mask.png", help="Filename suffix for predicted masks.")
    parser.add_argument("--gt-mask-suffix", default="_rgb_mask.png", help="Filename suffix for ground-truth masks.")
    parser.add_argument("--image-suffix", default=".jpg", help="Filename suffix for input frames.")
    parser.add_argument("--ignore-index", type=int, help="Optional ignore index override.")
    parser.add_argument("--background-index", type=int, help="Optional background index override.")
    parser.add_argument("--confidence-low-threshold", type=float, default=0.50, help="Low-confidence threshold.")
    parser.add_argument("--confidence-medium-threshold", type=float, default=0.80, help="Medium-confidence threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it.")
    parser.set_defaults(func=run_error_analysis_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Root entrypoint for SASVi experiment runs.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    add_run_subcommand(subparsers)
    add_error_analysis_subcommand(subparsers)

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
