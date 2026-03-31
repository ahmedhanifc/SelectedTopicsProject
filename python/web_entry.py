from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def numeric_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else -1, path.name)


def list_images(folder: Path, suffix_filter: str | None = None) -> list[Path]:
    files = []
    for entry in folder.iterdir():
        if not entry.is_file() or entry.suffix not in IMAGE_SUFFIXES:
            continue
        if suffix_filter is not None and not entry.name.endswith(suffix_filter):
            continue
        files.append(entry)
    return sorted(files, key=numeric_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple Python bridge for the Next.js prototype.")
    parser.add_argument("--input", required=True, help="Clip directory containing frames/ and masks/ subfolders.")
    args = parser.parse_args()

    clip_dir = Path(args.input).resolve()
    frames_dir = clip_dir / "frames"
    masks_dir = clip_dir / "masks"

    if not clip_dir.exists():
        print(json.dumps({"error": f"Input path does not exist: {clip_dir}"}), file=sys.stderr)
        return 1
    if not frames_dir.exists():
        print(json.dumps({"error": f"Frames directory does not exist: {frames_dir}"}), file=sys.stderr)
        return 1
    if not masks_dir.exists():
        print(json.dumps({"error": f"Masks directory does not exist: {masks_dir}"}), file=sys.stderr)
        return 1

    frame_files = list_images(frames_dir)
    mask_files = list_images(masks_dir)
    if not frame_files:
        print(json.dumps({"error": f"No frames found in {frames_dir}"}), file=sys.stderr)
        return 1
    if not mask_files:
        print(json.dumps({"error": f"No masks found in {masks_dir}"}), file=sys.stderr)
        return 1

    payload = {
        "clip": clip_dir.name,
        "frames_count": len(frame_files),
        "masks_count": len(mask_files),
        "mask_paths": [str(path.relative_to(clip_dir.parent.parent.parent)) for path in mask_files],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
