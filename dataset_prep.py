#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Pillow\nInstall with: pip install pillow"
    ) from exc


def frame_sort_key(path: Path) -> int:
    match = re.search(r"frame_(\d+)_endo\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected frame filename: {path}")
    return int(match.group(1))


def convert_clip(src_clip_dir: Path, dst_clip_dir: Path, image_format: str, overwrite: bool) -> int:
    frame_paths = sorted(src_clip_dir.glob("frame_*_endo.png"), key=frame_sort_key)

    if not frame_paths:
        print(f"Skipping {src_clip_dir}: no '*_endo.png' files found")
        return 0

    dst_clip_dir.mkdir(parents=True, exist_ok=True)

    for idx, src_img in enumerate(frame_paths, start=1):
        ext = ".jpg" if image_format.lower() == "jpg" else ".png"
        dst_img = dst_clip_dir / f"{idx:04d}{ext}"

        if dst_img.exists() and not overwrite:
            raise FileExistsError(f"File exists: {dst_img}. Use --overwrite to replace it.")

        if image_format.lower() == "jpg":
            with Image.open(src_img) as im:
                rgb = im.convert("RGB")
                rgb.save(dst_img, quality=95)
        else:
            shutil.copy2(src_img, dst_img)

    print(f"Converted {src_clip_dir.name}: {len(frame_paths)} frames -> {dst_clip_dir}")
    return len(frame_paths)


def find_clip_dirs(src_root: Path) -> list[Path]:
    clip_dirs = []
    for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        subdirs = sorted(p for p in video_dir.iterdir() if p.is_dir())
        for subdir in subdirs:
            if any(subdir.glob("frame_*_endo.png")):
                clip_dirs.append(subdir)
    return clip_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CholecSeg8k-style clip folders into SASVi-ready frame folders."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Path to dataset root containing folders like video01/video01_28660/",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Output root for SASVi-ready folders.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Optional single clip name to convert, e.g. video01_28660",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format. Use 'jpg' for SASVi as-is.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if not src_root.exists():
        raise SystemExit(f"Source root does not exist: {src_root}")

    clip_dirs = find_clip_dirs(src_root)

    if args.clip:
        clip_dirs = [p for p in clip_dirs if p.name == args.clip]
        if not clip_dirs:
            raise SystemExit(f"Clip not found: {args.clip}")

    if not clip_dirs:
        raise SystemExit("No clip folders found.")

    total_frames = 0
    for clip_dir in clip_dirs:
        dst_clip_dir = dst_root / clip_dir.name
        total_frames += convert_clip(
            src_clip_dir=clip_dir,
            dst_clip_dir=dst_clip_dir,
            image_format=args.format,
            overwrite=args.overwrite,
        )

    print(f"\nDone. Converted {len(clip_dirs)} clip(s), {total_frames} frame(s) total.")
    print(f"SASVi input root: {dst_root}")


if __name__ == "__main__":
    main()
