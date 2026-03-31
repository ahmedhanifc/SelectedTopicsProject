from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import DetrForSegmentation, Mask2FormerForUniversalSegmentation

CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.append(str(PYTHON_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from analysis_tools.inference_export import INFERENCE_METADATA_COLUMNS, write_rows_to_csv
from src.model import get_model_instance_segmentation
from src.sam2.eval_sasvi import (
    DETR,
    Mask2Former,
    MaskRCNN,
    nnUNet,
    sasvi_inference,
)
from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import (
    get_cataract1k_colormap,
)
from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import (
    get_cholecseg8k_colormap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run the prediction on (default: cuda)",
    )
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="sam2_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="./checkpoints/sam2_hiera_b+.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--overseer_checkpoint",
        type=str,
        required=True,
        help="path to the Overseer model checkpoint",
    )
    parser.add_argument(
        "--overseer_type",
        type=str,
        required=True,
        choices=["MaskRCNN", "DETR", "Mask2Former"],
        help="overseer model type",
    )
    parser.add_argument(
        "--nnunet_checkpoint",
        type=str,
        help="path to the nnUnet model checkpoint. If provided will use nnUnet prediction for tough cases.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["CADIS", "CHOLECSEG8K", "CATARACT1K"],
        help="dataset type to run the prediction on",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run SASVI prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing to the output masks",
    )
    parser.add_argument(
        "--save_binary_mask",
        action="store_true",
        help="whether to also save per object binary masks in addition to the combined mask",
    )
    parser.add_argument(
        "--overseer_mask_dir",
        type=str,
        help="directory to save predicted masks from overseer of each video.",
    )
    parser.add_argument(
        "--nnunet_mask_dir",
        type=str,
        help="directory to save predicted masks from nnunet of each video.",
    )
    parser.add_argument(
        "--analysis_output_dir",
        type=str,
        help="optional directory for confidence maps and inference metadata exported for analysis.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars for cleaner logs or timing-sensitive runs.",
    )
    parser.add_argument(
        "--metrics_output_path",
        type=Path,
        default=None,
        help="Optional path for the timing summary JSON. Defaults to <output_mask_dir>/timing_summary.json.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap on the number of videos for quick validation runs.",
    )
    return parser.parse_args()


def build_dataset_config(dataset_type: str, overseer_type: str) -> dict[str, object]:
    if dataset_type == "CADIS":
        return {
            "num_classes": 18,
            "ignore_indices": [255],
            "shift_by_1": True,
            "palette": get_cadis_colormap(),
            "maskrcnn_hidden_ft": 32,
            "maskrcnn_backbone": "ResNet18",
        }
    if dataset_type == "CHOLECSEG8K":
        return {
            "num_classes": 13,
            "ignore_indices": [0] if overseer_type == "Mask2Former" else [],
            "shift_by_1": False,
            "palette": get_cholecseg8k_colormap(),
            "maskrcnn_hidden_ft": 64,
            "maskrcnn_backbone": "ResNet50",
        }
    if dataset_type == "CATARACT1K":
        return {
            "num_classes": 14,
            "ignore_indices": [0] if overseer_type in {"DETR", "Mask2Former"} else [],
            "shift_by_1": False,
            "palette": get_cataract1k_colormap(),
            "maskrcnn_hidden_ft": 32,
            "maskrcnn_backbone": "ResNet18",
        }
    raise NotImplementedError


def build_overseer_model(args: argparse.Namespace, cfg: dict[str, object]):
    num_classes = int(cfg["num_classes"])
    ignore_indices = list(cfg["ignore_indices"])
    shift_by_1 = bool(cfg["shift_by_1"])

    if args.overseer_type == "MaskRCNN":
        model = get_model_instance_segmentation(
            num_classes=num_classes,
            trainable_backbone_layers=0,
            hidden_ft=int(cfg["maskrcnn_hidden_ft"]),
            custom_in_ft_box=None,
            custom_in_ft_mask=None,
            backbone=str(cfg["maskrcnn_backbone"]),
            img_size=(299, 299),
        )
        model.load_state_dict(torch.load(args.overseer_checkpoint, weights_only=True, map_location="cpu"))
        model = model.to(args.device)
        model.eval()
        return MaskRCNN(model, shift_by_1, ignore_indices, num_classes, device=args.device)

    if args.overseer_type == "DETR":
        num_train_classes = num_classes - len(ignore_indices) + 1
        model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=100,
        ).to(args.device)
        model.load_state_dict(torch.load(args.overseer_checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        return DETR(
            model,
            shift_by_1,
            ignore_indices,
            num_classes,
            num_train_classes,
            args.dataset_type,
            device=args.device,
        )

    if args.overseer_type == "Mask2Former":
        num_train_classes = num_classes - len(ignore_indices) + 1
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-instance",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=20,
        ).to(args.device)
        model.load_state_dict(torch.load(args.overseer_checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        return Mask2Former(
            model,
            shift_by_1,
            ignore_indices,
            num_classes,
            num_train_classes,
            args.dataset_type,
            device=args.device,
        )

    raise NotImplementedError


def maybe_build_nnunet(args: argparse.Namespace, ignore_indices: list[int]):
    if args.nnunet_checkpoint is None:
        return None

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    nnunet_predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=not args.disable_tqdm,
    )
    nnunet_predictor.initialize_from_trained_model_folder(
        args.nnunet_checkpoint,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    return nnUNet(nnunet_predictor, ignore_indices)


def list_video_names(base_video_dir: str) -> list[str]:
    video_names = [
        item
        for item in os.listdir(base_video_dir)
        if os.path.isdir(os.path.join(base_video_dir, item))
    ]
    return sorted(video_names)


def count_video_frames(base_video_dir: str, video_name: str) -> int:
    video_dir = os.path.join(base_video_dir, video_name)
    return len(
        [
            item
            for item in os.listdir(video_dir)
            if os.path.splitext(item)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    )


def main() -> None:
    args = parse_args()

    hydra_overrides_extra = ["++model.non_overlap_masks=true"]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    cfg = build_dataset_config(args.dataset_type, args.overseer_type)
    overseer_model = build_overseer_model(args, cfg)
    nnunet_model = maybe_build_nnunet(args, list(cfg["ignore_indices"]))

    video_names = list_video_names(args.base_video_dir)
    if args.max_videos is not None and args.max_videos > 0:
        video_names = video_names[: args.max_videos]

    print(f"running SASVI prediction on {len(video_names)} videos:\n{video_names}")

    overall_start = time.perf_counter()
    inference_rows: list[dict] = []
    video_summaries: list[dict[str, float | int | str]] = []

    video_iterator = tqdm(
        video_names,
        desc="eval_sasvi_timed",
        unit="video",
        disable=args.disable_tqdm,
    )
    for n_video, video_name in enumerate(video_iterator, start=1):
        frame_count = count_video_frames(args.base_video_dir, video_name)
        print(f"\n{n_video}/{len(video_names)} - running on {video_name}")
        start_time = time.perf_counter()
        inference_rows.extend(
            sasvi_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                overseer_type=args.overseer_type,
                overseer_mask_dir=args.overseer_mask_dir,
                overseer_model=overseer_model,
                nnunet_model=nnunet_model,
                num_classes=int(cfg["num_classes"]),
                ignore_indices=list(cfg["ignore_indices"]),
                shift_by_1=bool(cfg["shift_by_1"]),
                palette=cfg["palette"],
                dataset_type=args.dataset_type,
                score_thresh=args.score_thresh,
                save_binary_mask=args.save_binary_mask,
                analysis_output_dir=args.analysis_output_dir,
            )
        )
        elapsed_seconds = time.perf_counter() - start_time
        throughput_fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
        avg_frame_ms = 1000.0 * elapsed_seconds / frame_count if frame_count > 0 else 0.0
        summary = {
            "video_name": video_name,
            "frames": frame_count,
            "elapsed_seconds": elapsed_seconds,
            "throughput_fps": throughput_fps,
            "avg_frame_ms": avg_frame_ms,
        }
        video_summaries.append(summary)
        if not args.disable_tqdm:
            video_iterator.set_postfix(
                fps=f"{throughput_fps:.2f}",
                sec=f"{elapsed_seconds:.1f}",
            )

    if args.analysis_output_dir is not None and len(inference_rows) > 0:
        metadata_path = os.path.join(args.analysis_output_dir, "inference", "inference_metadata.csv")
        write_rows_to_csv(metadata_path, inference_rows, INFERENCE_METADATA_COLUMNS)

    overall_elapsed = time.perf_counter() - overall_start
    total_frames = sum(int(item["frames"]) for item in video_summaries)
    overall_fps = total_frames / overall_elapsed if overall_elapsed > 0 else 0.0

    summary_payload = {
        "base_video_dir": args.base_video_dir,
        "output_mask_dir": args.output_mask_dir,
        "dataset_type": args.dataset_type,
        "overseer_type": args.overseer_type,
        "device": args.device,
        "video_count": len(video_summaries),
        "total_frames": total_frames,
        "elapsed_seconds": overall_elapsed,
        "throughput_fps": overall_fps,
        "videos": video_summaries,
    }
    summary_path = args.metrics_output_path or (Path(args.output_mask_dir) / "timing_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Time taken: {overall_elapsed:.6f} seconds")
    print(f"End-to-end throughput: {overall_fps:.3f} FPS")
    print(f"Timing summary: {summary_path}")
    print(
        f"completed SASVI prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
