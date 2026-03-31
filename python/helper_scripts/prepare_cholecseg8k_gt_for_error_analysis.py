#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


FRAME_PATTERN = re.compile(r"frame_(\d+)_endo_watershed_mask\.png$")


def frame_sort_key(path: Path) -> int:
    match = FRAME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Unexpected ground-truth filename: {path}")
    return int(match.group(1))


def find_clip_dirs(src_root: Path) -> list[Path]:
    clip_dirs: list[Path] = []
    for video_dir in sorted(path for path in src_root.iterdir() if path.is_dir()):
        for clip_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
            if any(clip_dir.glob("frame_*_endo_watershed_mask.png")):
                clip_dirs.append(clip_dir)
    return clip_dirs


def export_clip_gt(src_clip_dir: Path, dst_clip_dir: Path, overwrite: bool, symlink: bool) -> int:
    gt_paths = sorted(src_clip_dir.glob("frame_*_endo_watershed_mask.png"), key=frame_sort_key)
    if not gt_paths:
        print(f"Skipping {src_clip_dir}: no '*_endo_watershed_mask.png' files found")
        return 0

    dst_clip_dir.mkdir(parents=True, exist_ok=True)

    for idx, src_gt in enumerate(gt_paths, start=1):
        dst_gt = dst_clip_dir / f"{idx:04d}_rgb_mask.png"
        if dst_gt.exists() or dst_gt.is_symlink():
            if not overwrite:
                raise FileExistsError(f"File exists: {dst_gt}. Use --overwrite to replace it.")
            if dst_gt.is_dir():
                raise IsADirectoryError(f"Expected file path but found directory: {dst_gt}")
            dst_gt.unlink()

        if symlink:
            dst_gt.symlink_to(src_gt.resolve())
        else:
            shutil.copy2(src_gt, dst_gt)

    print(f"Prepared {src_clip_dir.name}: {len(gt_paths)} GT mask(s) -> {dst_clip_dir}")
    return len(gt_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CholecSeg8k ground-truth masks for SASVi error analysis."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Path to the original CholecSeg8k dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Output root that will be used as --gt_root for error analysis.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Optional single clip name to export, e.g. video01_28660.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination root.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files.",
    )
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if not src_root.exists():
        raise SystemExit(f"Source root does not exist: {src_root}")

    clip_dirs = find_clip_dirs(src_root)
    if args.clip:
        clip_dirs = [path for path in clip_dirs if path.name == args.clip]
        if not clip_dirs:
            raise SystemExit(f"Clip not found: {args.clip}")

    if not clip_dirs:
        raise SystemExit("No clip folders with ground-truth masks found.")

    total_masks = 0
    for clip_dir in clip_dirs:
        total_masks += export_clip_gt(
            src_clip_dir=clip_dir,
            dst_clip_dir=dst_root / clip_dir.name,
            overwrite=args.overwrite,
            symlink=args.symlink,
        )

    print(f"\nDone. Prepared {len(clip_dirs)} clip(s), {total_masks} mask(s) total.")
    print(f"Use this as --gt_root: {dst_root}")


if __name__ == "__main__":
    main()
