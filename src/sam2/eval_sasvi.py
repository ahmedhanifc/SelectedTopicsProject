# Using nnunet on top of overseer for tough instances. But not part of original paper and can be left out
#TODO: Prediction on reverse direction when overseer prediction confidence is high
#TODO: Fine-tune SAM2 on the dataset

import argparse
import itertools
import os
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import measure
from scipy.ndimage import find_objects
from scipy.spatial.distance import cdist
import albumentations as A
import torchvision.transforms as T
from transformers import DetrImageProcessor, DetrForSegmentation, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

try:
    from sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation import get_cholecseg8k_colormap
    from sds_playground.datasets.cadisv2.cadisv2_visualisation import get_cadis_colormap
    from sds_playground.datasets.cataract1k.cataract1ksegm_visualisation import get_cataract1k_colormap
except ModuleNotFoundError:
    def _fallback_colormap(num_classes, dataset_type=None):
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

    def get_cholecseg8k_colormap():
        return _fallback_colormap(13, dataset_type="CHOLECSEG8K")

    def get_cadis_colormap():
        return _fallback_colormap(18)

    def get_cataract1k_colormap():
        return _fallback_colormap(14)

# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sam2.build_sam import build_sam2_video_predictor
from analysis_tools.inference_export import (
    INFERENCE_METADATA_COLUMNS,
    compute_confidence_map,
    save_confidence_map,
    summarise_confidence_map,
    write_rows_to_csv,
    write_markdown_report,
)
from analysis_tools.config import get_dataset_config
from analysis_tools.error_analysis import compute_frame_metrics, load_dataset_mask


def configure_cpu_runtime():
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass

    cpu_threads = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
from src.sam2_utils import kmeans_sampling
from src.data import remap_labels, insert_component_masks
from src.utils import process_detr_outputs, process_mask2former_outputs
from src.model import get_model_instance_segmentation


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item) 
        else:
            yield item 


def load_ann_png(path, 
                shift_by_1,
                reshape_size=None, 
                convert_to_label=False, 
                palette=None
                ):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    if reshape_size is not None:
        mask = mask.resize(reshape_size, Image.NEAREST)
    mask = np.array(mask).astype(np.uint8)

    if convert_to_label:
        if palette is None:
            raise ValueError("palette is required to convert the mask to label")
        else:
            mask_reshaped = mask.reshape(-1, 3)
            #TODO: Need to handle ignore and background better
            if shift_by_1:
                output = np.full(mask_reshaped.shape[0], 255, dtype=np.uint8)
            else:
                output = np.full(mask_reshaped.shape[0], 0, dtype=np.uint8)
            for i, color in enumerate(palette):
                matches = np.all(mask_reshaped == color, axis=1)
                output[matches] = i
            mask = output.reshape(mask.shape[:2])
    return mask


def load_ann_npz(path, 
                 reshape_size=None,
                 ignore_indices=None,
                 ):
    """Load a NPZ file as a mask."""
    npz_mask = np.load(path, allow_pickle=True)
    npz_mask = npz_mask["arr"].astype(bool)
    if reshape_size is not None:
        mask = np.array([resize(mask, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for mask in npz_mask])
    else:
        mask = npz_mask.astype(bool)
    #TODO: Need to handle ignore and background better
    if len(ignore_indices) > 0:
        length = len(ignore_indices)
        mask = mask[:-length]
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
    elif len(ignore_indices) == 0:
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
        object_ids = object_ids[1:]

    per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}
    return per_obj_mask


def save_ann_png(path, mask, palette, reshape_size=None):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    if reshape_size is not None:
        output_mask = output_mask.resize(reshape_size, Image.NEAREST)
    output_mask.putpalette(np.uint8(palette))
    output_mask.save(path)


def get_per_obj_mask(mask_path, frame_name, use_binary_mask, width, height, ignore_indices, shift_by_1, palette):
    """Split a mask into per-object masks."""
    # For CholecSeg8K, the gt mask and frame are saved according to frame rate. But overseer paths are not according to frame rate.
    if not use_binary_mask:
        mask = load_ann_png(path=os.path.join(mask_path, f"{frame_name}_rgb_mask.png"),
                            shift_by_1=shift_by_1,
                            reshape_size=(width, height),
                            palette=palette)
        object_ids = np.unique(mask)
        object_ids = np.array([item for item in object_ids if item not in ignore_indices])
        object_ids = object_ids[object_ids >= 0].tolist()
        per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    else:
        if width is not None and height is not None: reshape_size=(height, width)
        else: reshape_size = None

        per_obj_mask = load_ann_npz(path=os.path.join(mask_path, f"{frame_name}_binary_mask.npz"), 
                                        reshape_size=reshape_size, 
                                        ignore_indices=ignore_indices)
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width, shift_by_1):
    """Combine per-object masks into a single mask."""
    #TODO: Need to handle ignore and background better 
    if shift_by_1:
        mask = np.full((height, width), 255, dtype=np.uint8)
    else:  
        mask = np.full((height, width), 0, dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def get_bbox_from_mask(mask):
    """Get the bounding box from gt mask."""
    unique_labels = np.unique(mask)
    for label in unique_labels:
        label = int(label)
        binary_mask = (mask == label)
        # Get the bounding box
        slices = find_objects(binary_mask)[0]
        bounding_box = (slices[0].start, slices[0].stop, slices[1].start, slices[1].stop)
        bbox = {label: bounding_box}
    return bbox


def get_points_from_mask(mask, label_ids=None, num_points=20):
    """Get the points from gt mask."""
    label_ids = label_ids if label_ids is not None else set(mask.keys())
    for label in label_ids:
        binary_mask = mask[label]
        # Get the prompt
        try:
            points = kmeans_sampling(torch.tensor(np.argwhere(binary_mask)), num_points)
            points = points.cpu().numpy().tolist()
            points = [[y, x] for x, y in points]
        except:
            points = []
        prompt_points = {label: points}
    return prompt_points


def infer_frame_sort_key(stem):
    if stem.startswith("frame_"):
        stem = stem[6:]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Unable to parse frame index from '{stem}'")


def list_video_frames(video_dir):
    frame_infos = []
    for filename in os.listdir(video_dir):
        stem, ext = os.path.splitext(filename)
        if ext not in IMAGE_EXTENSIONS:
            continue
        if "_mask" in stem:
            continue
        frame_infos.append({
            "stem": stem,
            "ext": ext,
            "path": os.path.join(video_dir, filename),
        })

    frame_infos.sort(key=lambda item: infer_frame_sort_key(item["stem"]))
    return frame_infos


def discover_video_dirs(base_video_dir):
    discovered = []
    for entry in sorted(os.listdir(base_video_dir)):
        entry_path = os.path.join(base_video_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        if list_video_frames(entry_path):
            discovered.append((entry, entry_path))
            continue

        for child in sorted(os.listdir(entry_path)):
            child_path = os.path.join(entry_path, child)
            if os.path.isdir(child_path) and list_video_frames(child_path):
                discovered.append((child, child_path))
    return discovered


def build_frame_subset_dir(video_name, frame_infos, output_root=None):
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)
    subset_root = tempfile.mkdtemp(prefix=f"sasvi_{video_name}_", dir=output_root)
    subset_dir = os.path.join(subset_root, video_name)
    os.makedirs(subset_dir, exist_ok=True)
    for frame_info in frame_infos:
        dst_path = os.path.join(subset_dir, f"{frame_info['stem']}{frame_info['ext']}")
        os.symlink(os.path.abspath(frame_info["path"]), dst_path)
    return subset_root, subset_dir


GT_MASK_SUFFIX_CANDIDATES = (
    "_watershed_mask.png",
    "_color_mask.png",
    "_mask.png",
)


def resolve_gt_mask_path(frame_name, video_dir, video_name, base_video_dir, gt_root_dir=None):
    candidate_roots = []
    if gt_root_dir is not None:
        relative_video_dir = os.path.relpath(video_dir, start=base_video_dir)
        candidate_roots.append(os.path.join(gt_root_dir, relative_video_dir))
        candidate_roots.append(os.path.join(gt_root_dir, video_name))
    candidate_roots.append(video_dir)

    seen_paths = set()
    for root in candidate_roots:
        for suffix in GT_MASK_SUFFIX_CANDIDATES:
            candidate = os.path.abspath(os.path.join(root, f"{frame_name}{suffix}"))
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            if os.path.isfile(candidate):
                return candidate
    return None


def ensure_cached_overseer_prediction(cache, frame_idx, frame_names, frame_path_by_name, overseer_model, reshape_size):
    if frame_idx not in cache:
        frame_name = frame_names[frame_idx]
        cache[frame_idx] = overseer_model.get_prediction(
            [frame_path_by_name[frame_name]],
            reshape_size=reshape_size,
        )
    return cache[frame_idx]


def per_obj_mask_to_label_map(per_obj_mask, height, width, shift_by_1):
    if not per_obj_mask:
        fill_value = 255 if shift_by_1 else 0
        return np.full((height, width), fill_value, dtype=np.uint8)
    return put_per_obj_mask(per_obj_mask, height, width, shift_by_1)


def per_obj_mask_to_foreground(per_obj_mask, height, width):
    foreground = np.zeros((height, width), dtype=bool)
    for object_mask in per_obj_mask.values():
        foreground |= np.asarray(object_mask, dtype=bool).reshape(height, width)
    return foreground


def compute_binary_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_classwise_iou(per_obj_mask_a, per_obj_mask_b, height, width):
    labels = sorted(set(per_obj_mask_a) | set(per_obj_mask_b))
    if not labels:
        return 1.0, {}

    per_class_iou = {}
    empty_mask = np.zeros((height, width), dtype=bool)
    for label in labels:
        mask_a = np.asarray(per_obj_mask_a.get(label, empty_mask), dtype=bool).reshape(height, width)
        mask_b = np.asarray(per_obj_mask_b.get(label, empty_mask), dtype=bool).reshape(height, width)
        per_class_iou[label] = compute_binary_iou(mask_a, mask_b)

    mean_iou = float(np.mean(list(per_class_iou.values())))
    return mean_iou, per_class_iou


def compute_boundary_distance(mask_a, mask_b):
    mask_a = np.asarray(mask_a, dtype=np.uint8)
    mask_b = np.asarray(mask_b, dtype=np.uint8)
    if mask_a.sum() == 0 and mask_b.sum() == 0:
        return 0.0
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("inf")

    contours_a = measure.find_contours(mask_a, level=0.5)
    contours_b = measure.find_contours(mask_b, level=0.5)
    if not contours_a and not contours_b:
        return 0.0
    if not contours_a or not contours_b:
        return float("inf")

    points_a = np.concatenate(contours_a, axis=0)
    points_b = np.concatenate(contours_b, axis=0)
    distances = cdist(points_a, points_b)
    symmetric_distance = 0.5 * (
        distances.min(axis=1).mean() + distances.min(axis=0).mean()
    )
    return float(symmetric_distance)


def compute_disagreement_metrics(per_obj_sam_mask, per_obj_overseer_mask, height, width):
    mean_iou, per_class_iou = compute_classwise_iou(
        per_obj_mask_a=per_obj_sam_mask,
        per_obj_mask_b=per_obj_overseer_mask,
        height=height,
        width=width,
    )
    sam_foreground = per_obj_mask_to_foreground(per_obj_sam_mask, height, width)
    overseer_foreground = per_obj_mask_to_foreground(per_obj_overseer_mask, height, width)
    foreground_iou = compute_binary_iou(sam_foreground, overseer_foreground)
    return {
        "iou": mean_iou,
        "foreground_iou": foreground_iou,
        "per_class_iou": per_class_iou,
        "sam_foreground": sam_foreground,
        "overseer_foreground": overseer_foreground,
    }


def save_disagreement_visual(path, frame_path, sam_foreground, overseer_foreground):
    frame = np.array(Image.open(frame_path).convert("RGB"))
    sam_only = np.logical_and(sam_foreground, ~overseer_foreground)
    overseer_only = np.logical_and(overseer_foreground, ~sam_foreground)
    overlap = np.logical_and(sam_foreground, overseer_foreground)

    overlay = frame.copy()
    overlay[sam_only] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[overseer_only] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[overlap] = np.array([255, 255, 0], dtype=np.uint8)

    Image.fromarray(overlay).save(path)


def get_primary_visual_palette(num_classes):
    base_colors = np.array([
        [127, 127, 127],  # background
        [255, 0, 0],      # red
        [0, 0, 255],      # blue
        [0, 200, 0],      # green
        [255, 255, 0],    # yellow
        [255, 0, 255],    # magenta
        [0, 255, 255],    # cyan
        [255, 128, 0],    # orange
        [128, 0, 255],    # violet
        [139, 69, 19],    # brown
        [255, 255, 255],  # white
        [0, 0, 0],        # black
        [0, 128, 128],    # teal
        [128, 128, 0],    # olive
        [128, 0, 0],      # maroon
        [0, 128, 0],      # dark green
        [0, 128, 255],    # azure
        [255, 0, 128],    # rose
        [64, 64, 255],    # strong periwinkle
        [0, 180, 120],    # sea green
    ], dtype=np.uint8)
    if num_classes <= len(base_colors):
        return base_colors[:num_classes]
    extra = []
    for idx in range(len(base_colors), num_classes):
        extra.append([(53 * idx) % 256, (97 * idx) % 256, (193 * idx) % 256])
    return np.vstack([base_colors, np.array(extra, dtype=np.uint8)])


def smooth_label_map(label_map, fill_value):
    smoothed = np.full(label_map.shape, fill_value, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    labels = [label for label in np.unique(label_map).tolist() if label != fill_value]
    for label in labels:
        binary = (label_map == label).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        smoothed[binary.astype(bool)] = label
    return smoothed


def export_mask_variants(
    output_mask_dir,
    video_name,
    frame_name,
    raw_mask,
    smoothed_mask,
    output_palette,
):
    variant_masks = {
        "raw": raw_mask,
        "smoothed": smoothed_mask,
    }
    for variant_name, variant_mask in variant_masks.items():
        variant_dir = os.path.join(output_mask_dir, variant_name, video_name)
        os.makedirs(variant_dir, exist_ok=True)
        save_ann_png(
            os.path.join(variant_dir, f"{frame_name}_rgb_mask.png"),
            variant_mask,
            output_palette,
        )


def label_map_to_binary_stack(label_map, num_classes, shift_by_1):
    binary_stack = np.zeros((num_classes, *label_map.shape), dtype=bool)
    for class_idx in range(num_classes):
        binary_stack[class_idx] = label_map == class_idx
    if shift_by_1:
        binary_stack[-1] |= label_map == 255
    return binary_stack


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    output_palette,
    save_binary_mask,
    num_classes,
    shift_by_1,
    vis_mode=False,
):
    """Save masks to a directory as PNG files."""
    per_obj_output_mask = {
        key: np.expand_dims(value, axis=0) if value.ndim == 2 else value
        for key, value in per_obj_output_mask.items()
    }
    raw_output_mask = put_per_obj_mask(per_obj_output_mask, height, width, shift_by_1)
    fill_value = 255 if shift_by_1 else 0
    smoothed_output_mask = smooth_label_map(raw_output_mask, fill_value=fill_value)

    export_mask_variants(
        output_mask_dir=output_mask_dir,
        video_name=video_name,
        frame_name=frame_name,
        raw_mask=raw_output_mask,
        smoothed_mask=smoothed_output_mask,
        output_palette=output_palette,
    )

    if save_binary_mask:
        for i in range(num_classes):
            if i not in per_obj_output_mask:
                per_obj_output_mask[i] = np.full((1, height, width), False)
        output_mask = dict(sorted(per_obj_output_mask.items()))
        output_mask = np.array(list(output_mask.values()))
        output_mask = np.squeeze(output_mask, axis=None)
        #TODO: Need to handle ignore and background better
        if shift_by_1:
            output_mask[-1] |= ~output_mask[:-1].any(axis=0)
        else:
            output_mask[0] |= ~output_mask[1:].any(axis=0)

        raw_binary_dir = os.path.join(output_mask_dir, "raw", video_name)
        smoothed_binary_dir = os.path.join(output_mask_dir, "smoothed", video_name)
        os.makedirs(raw_binary_dir, exist_ok=True)
        os.makedirs(smoothed_binary_dir, exist_ok=True)
        raw_binary = label_map_to_binary_stack(raw_output_mask, num_classes, shift_by_1)
        smoothed_binary = label_map_to_binary_stack(smoothed_output_mask, num_classes, shift_by_1)
        np.savez_compressed(file=os.path.join(raw_binary_dir, f"{frame_name}_binary_mask.npz"), arr=raw_binary)
        np.savez_compressed(file=os.path.join(smoothed_binary_dir, f"{frame_name}_binary_mask.npz"), arr=smoothed_binary)

        if vis_mode:
            rows, cols = 4, 4  # Adjust based on the number of masks
            fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                if i < output_mask.shape[0]:
                    ax.imshow(output_mask[i], cmap='gray')  # Display each mask
                    ax.set_title(f"Mask {i+1}")
                ax.axis('off')
            for i in range(output_mask.shape[0], len(axes)):
                fig.delaxes(axes[i])
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(output_mask_dir, "raw", video_name, f"{frame_name}_visualization.jpg"), bbox_inches='tight', pad_inches=0.1)
            plt.close()


def get_unique_label(per_obj_mask_n):
    unique_lable_n = []
    for per_obj_mask in per_obj_mask_n:
        unique_lable_n.append(sorted(set(per_obj_mask.keys())))
    return unique_lable_n


def choose_duplicate_label(per_obj_mask_n, duplicate_label):
    selections = {}
    for per_obj_mask in per_obj_mask_n:
        unique_label = sorted(set(per_obj_mask.keys()))

        # if only one label from duplicate exist in next frame
        matching = list(set(duplicate_label) & set(unique_label))
        if len(matching) == 0:
            continue
        if len(matching) == 1:
            mask_size = np.sum((per_obj_mask[matching[0]] == 1))
            if matching[0] in selections:
                selections[matching[0]] += mask_size
            else: selections[matching[0]] = mask_size
        # if both or more labels from duplicate exist in next frame 
        else:
            temp_selections = []
            for pair in list(itertools.combinations(matching, 2)):
                mask1 = per_obj_mask[pair[0]]
                mask2 = per_obj_mask[pair[1]]
                true_positions_1 = (mask1 == 1)                                                                                                                
                true_positions_2 = (mask2 == 1)
                matching_true_positions = np.logical_and(true_positions_1, true_positions_2)
                similarity_1 = np.sum(matching_true_positions) / np.sum(true_positions_1)
                similarity_2 = np.sum(matching_true_positions) / np.sum(true_positions_2)

                if similarity_1 >= 0.70 and similarity_2 >= 0.70:
                    temp_selections.append(pair[0])
                    temp_selections.append(pair[1])
            
            for item in list(dict.fromkeys(temp_selections)):
                mask_size = np.sum((per_obj_mask[item] == 1))
                if item in selections:
                    selections[item] += mask_size
                else: selections[item] = mask_size

    # set flag true if highest n labels are close together. Will opt to use nnunet for such cases
    best_key = max(selections, key=selections.get)
    best_score = selections[best_key]
    use_nnunet = False
    for key in selections:
        if key != best_key:
            if selections[key] >= best_score * 0.95:
                use_nnunet = True

    return selections, use_nnunet


class nnUNet():
    def __init__(self, model, ignore_indices):
        self.model = model
        self.ignore_indices = ignore_indices

    def get_prediction(self, image_path, reshape_size=None):
        image, props = NaturalImage2DIO().read_images([image_path])
        image = image.squeeze(1).transpose(1, 2, 0)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        image = image.transpose(2, 0, 1).reshape(3, 1, 128, 128)
        pred_mask = self.model.predict_single_npy_array(image, props, None, None, False)
        pred_mask = pred_mask.squeeze(0)

        if reshape_size is not None:
            pred_mask = cv2.resize(pred_mask, reshape_size, interpolation=cv2.INTER_NEAREST)

        object_ids = np.unique(pred_mask)
        object_ids = object_ids[object_ids >= 0].tolist()

        length = len(self.ignore_indices)
        if length > 0:
            object_ids = object_ids[:-length]
            per_obj_mask = {object_id: (pred_mask == object_id) for object_id in object_ids}
        elif length == 0:
            object_ids = object_ids[1:]
            per_obj_mask = {object_id: (pred_mask == object_id) for object_id in object_ids}
        return per_obj_mask


class MaskRCNN():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, device="cuda"):
        self.model = model
        self.transform =  T.Compose([T.ToTensor(), T.Resize((299, 299)),T.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.device = device

    def get_prediction(self, image_list, reshape_size=None):
        images = [np.array(Image.open(i).convert('RGB')) for i in image_list]
        images = [self.transform(i).to(self.device) for i in images]
        with torch.no_grad():
            predictions = self.model(images)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        binary_pred_mask = (pred_mask > 0.5).int()
        
        padded_binary_pred_mask = insert_component_masks(
            binary_pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]
                
        # for cholec and cat1k, there are background class and also area with no prediction
        # for cadis, only no prediction area.  
        if self.shift_by_1:
            background_class = self.num_classes-1
            object_ids = object_ids[object_ids != background_class]
        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}
        return per_obj_mask
        

class Mask2Former():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, num_train_classes, dataset_type, device="cuda"):
        self.model = model
        self.transform =  A.Compose([A.Resize(*(299, 299)), A.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.num_train_classes = num_train_classes
        self.dataset_type = dataset_type
        self.device = device
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance", reduce_labels=True)

    def get_prediction(self, image_list, reshape_size=(299,299)):
        images = [np.array(Image.open(i)) for i in image_list]
        images = [self.transform(image=i)['image'] for i in images]
        images = [A.pytorch.functional.img_to_tensor(i).to(self.device) for i in images]
        images = torch.stack(images)
        # images = self.processor(images=[image.cpu().numpy() for image in images],
        #                         do_rescale=False,
        #                         return_tensors="pt").to(self.device)

        with torch.no_grad():
            predictions = self.model(images)
        predictions = process_mask2former_outputs(predictions,
                                           image_size=reshape_size,
                                           num_labels=self.num_train_classes,
                                           threshold=0.0) #if self.dataset_type == "CADIS" else 0.5)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        
        #TODO: All this dataset specific conditioning need to be more cleaner
        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            pred_labels -= 1

        # CAT1K: i think just needs to remove ignore class 0
        if self.dataset_type == "CATARACT1K":
            self.ignore_indices = []  

        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        
        padded_binary_pred_mask = insert_component_masks(
            pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}

        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 0}

        # CADIS: need to remove last class (remaining 0 to 16) (ignore is 17. but actually 255)
        if self.dataset_type == "CADIS":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 17}
        return per_obj_mask
    

class DETR():
    def __init__(self, model, shift_by_1, ignore_indices, num_classes, num_train_classes, dataset_type, device="cuda"):
        self.model = model
        self.transform =  A.Compose([A.Resize(*(200, 200)), A.Normalize(0.0, 1.0)])
        self.shift_by_1 = shift_by_1
        self.ignore_indices = ignore_indices
        self.num_classes = num_classes
        self.num_train_classes = num_train_classes
        self.dataset_type = dataset_type
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")

    def get_prediction(self, image_list, reshape_size=(299,299)):
        images = [np.array(Image.open(i)) for i in image_list]
        images = [self.transform(image=i)['image'] for i in images]
        images = [A.pytorch.functional.img_to_tensor(i).to(self.device) for i in images]
        images = self.processor(images=[image.cpu().numpy() for image in images],
                                do_rescale=False,
                                return_tensors="pt").to(self.device)

        with torch.no_grad():
            predictions = self.model(**images)
        predictions = process_detr_outputs(predictions,
                                           image_size=reshape_size,
                                           num_labels=self.num_train_classes,
                                           threshold=0.0) #if self.dataset_type == "CADIS" else 0.5)

        pred_mask = predictions[0]["masks"]
        pred_labels = predictions[0]["labels"]
        
        #TODO: All this dataset specific conditioning need to be more cleaner
        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            pred_labels -= 1

        # CAT1K: i think just needs to remove ignore class 0
        if self.dataset_type == "CATARACT1K":
            self.ignore_indices = []  

        if self.shift_by_1: pred_labels -= 1
        remapped_pred_labels = remap_labels(pred_labels, self.num_classes, self.ignore_indices)
        
        padded_binary_pred_mask = insert_component_masks(
            pred_mask,
            remapped_pred_labels,
            self.num_classes,
            ignore_index=self.ignore_indices[0] if len(self.ignore_indices) != 0 else None)
        mask = padded_binary_pred_mask.to(torch.bool).cpu().numpy()

        if reshape_size is not None:
            mask = np.array([resize(i, reshape_size, order=0, preserve_range=True, anti_aliasing=False).astype(bool) for i in mask])
        object_ids = np.where(np.any(mask, axis=(1, 2)))[0]        
        per_obj_mask = {object_id: mask[object_id] for object_id in object_ids}

        # CHOLEC: need to enable this and remove class 0
        if self.dataset_type == "CHOLECSEG8K":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 0}

        # CADIS: need to remove last class (remaining 0 to 16) (ignore is 17. but actually 255)
        if self.dataset_type == "CADIS":
            per_obj_mask = {object_id: mask for object_id, mask in per_obj_mask.items() if object_id != 17}
        return per_obj_mask
            

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def sasvi_inference(
    predictor,
    base_video_dir,
    output_mask_dir,
    video_name,
    overseer_type,
    overseer_mask_dir,
    overseer_model,
    nnunet_model,

    num_classes,
    ignore_indices,
    shift_by_1,
    palette,
    dataset_type,
    gt_dataset_config,
    video_dir=None,
    gt_root_dir=None,
    start_frame=None,
    end_frame=None,
    frame_name=None,

    score_thresh=0.0,
    save_binary_mask=False,
    analysis_output_dir=None,
    enable_disagreement_gate=False,
    disagreement_iou_threshold=0.5,
    disagreement_bad_frames=2,
    enable_boundary_distance_gate=False,
    boundary_distance_threshold=20.0,
    save_disagreement_visuals=False,
    max_disagreement_visuals=10,
):
    """Run inference on a single video with the given predictor."""
    video_dir = video_dir or os.path.join(base_video_dir, video_name)
    frame_infos = list_video_frames(video_dir)
    if not frame_infos:
        raise RuntimeError(f"No input frames found in {video_dir}")

    if frame_name is not None:
        frame_infos = [info for info in frame_infos if info["stem"] == frame_name]
        if not frame_infos:
            raise RuntimeError(f"Frame '{frame_name}' not found in {video_dir}")

    if start_frame is not None:
        frame_infos = [info for info in frame_infos if infer_frame_sort_key(info["stem"]) >= start_frame]
    if end_frame is not None:
        frame_infos = [info for info in frame_infos if infer_frame_sort_key(info["stem"]) <= end_frame]
    if not frame_infos:
        raise RuntimeError(f"No frames left to process for {video_name} after filtering.")

    frame_names = [info["stem"] for info in frame_infos]
    frame_path_by_name = {info["stem"]: info["path"] for info in frame_infos}
    predictor_video_dir = video_dir
    temp_subset_root = None
    subset_mode = (
        frame_name is not None
        or start_frame is not None
        or end_frame is not None
    )
    if subset_mode:
        temp_subset_root, predictor_video_dir = build_frame_subset_dir(
            video_name=video_name,
            frame_infos=frame_infos,
            output_root=analysis_output_dir,
        )

    per_obj_input_mask_cached = {}
    
    # load the video frames and initialize the inference state of SAM2 on this video
    inference_state = predictor.init_state(
        video_path=predictor_video_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    break_endless_loop = False
    inference_rows = {}
    confidence_dir = None
    disagreement_visual_dir = None
    if analysis_output_dir is not None:
        confidence_dir = os.path.join(analysis_output_dir, "inference", "confidence_maps", video_name)
        os.makedirs(confidence_dir, exist_ok=True)
        if save_disagreement_visuals:
            disagreement_visual_dir = os.path.join(
                analysis_output_dir, "inference", "disagreement_visuals", video_name
            )
            os.makedirs(disagreement_visual_dir, exist_ok=True)

    raw_output_mask_dir = os.path.join(output_mask_dir, "raw")
    smoothed_output_mask_dir = os.path.join(output_mask_dir, "smoothed")
    os.makedirs(os.path.join(raw_output_mask_dir, video_name), exist_ok=True)
    os.makedirs(os.path.join(smoothed_output_mask_dir, video_name), exist_ok=True)

    trace_rows = []
    video_summary = {
        "video_name": video_name,
        "frames_processed": 0,
        "segments_started": 0,
        "class_change_reprompts": 0,
        "disagreement_reprompts": 0,
        "mean_iou": 1.0,
        "min_iou": 1.0,
        "gt_frames_evaluated": 0,
        "gt_macro_iou_mean": None,
        "gt_macro_dice_mean": None,
        "gt_pixel_accuracy_mean": None,
    }
    per_frame_ious = []
    gt_macro_ious = []
    gt_macro_dices = []
    gt_pixel_accuracies = []
    prompt_label_list = []
    negative_duplicate_list = []
    old_label = []
    current_label = []
    break_endless_loop = False
    disagreement_visual_count = 0
    idx = 0
    while idx < len(frame_names):
        segment_id = video_summary["segments_started"] + 1
        video_summary["segments_started"] += 1
        segment_start_idx = idx
        disagreement_counter = 0

        # clear the prompts from previous runs
        print(f"[segment] video={video_name} segment={segment_id} start_frame={frame_names[idx]} idx={idx}")
        predictor.reset_state(inference_state=inference_state)
        buffer_length = 25

        # clean up the cached mask if it gets too big. Slightly larger than future_n_frame length
        MAX_CACHE_SIZE = 20
        while len(per_obj_input_mask_cached) > MAX_CACHE_SIZE:
            smallest_idx = min(per_obj_input_mask_cached.keys())
            del per_obj_input_mask_cached[smallest_idx]

        # this loads the masks. will add those input masks to SAM 2 inference state before propagations
        per_obj_input_mask = ensure_cached_overseer_prediction(
            cache=per_obj_input_mask_cached,
            frame_idx=idx,
            frame_names=frame_names,
            frame_path_by_name=frame_path_by_name,
            overseer_model=overseer_model,
            reshape_size=(height, width),
        )

        if idx > 0:
            # to make it more stable, if something ignored in overseer frame, but detected in previous sam2 frame, add it to the prompt mask
            # but didnt work for cat1k and cholec80 bcs it slowly covers the background which should have no label
            if dataset_type == "CADIS":
                per_obj_previous_mask = get_per_obj_mask(
                                            mask_path=os.path.join(raw_output_mask_dir, video_name), 
                                            frame_name=frame_names[idx],
                                            use_binary_mask=False, 
                                            width=width, 
                                            height=height,
                                            ignore_indices=ignore_indices, 
                                            shift_by_1=shift_by_1, 
                                            palette=palette)
            
                ignore_class_input_mask = ~np.array(list(per_obj_input_mask.values())).any(axis=0)
                ignore_class_previous_mask = ~np.array(list(per_obj_previous_mask.values())).any(axis=0)
                additional_points = np.argwhere(ignore_class_input_mask & ~ignore_class_previous_mask)
                merged_previous_mask = put_per_obj_mask(per_obj_previous_mask, height, width, shift_by_1)
                for pos in additional_points:
                    val = merged_previous_mask[tuple(pos)]
                    if val in list(set(old_label) & set(current_label)):
                        per_obj_input_mask[val][tuple(pos)] = True

            for obj_id in prompt_label_list:
                # adding positive points for new objects
                selected_point = get_points_from_mask(per_obj_input_mask, label_ids=[obj_id])
                if selected_point[obj_id]:
                    points = np.array(selected_point[obj_id], dtype=np.float32)
                    labels = np.ones((len(selected_point[obj_id]),), dtype=np.int32) 
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )

                # since mask have inaccurate label at positions where new tool is occupied, modify those areas to be false        
                false_positions = np.argwhere(per_obj_input_mask[obj_id])
                for obj_input_mask in per_obj_input_mask:
                    # sometimes sam2 is already correctly guessing them, so we dont want to disable those areas
                        if obj_input_mask != obj_id:
                            for false_pos in false_positions:
                                per_obj_input_mask[obj_input_mask][tuple(false_pos)] = False

            # update the old label 
            yet_another_unique_label = old_label
            break_endless_loop = True
            negative_duplicate_list = [x for xs in negative_duplicate_list for x in xs]
            old_label = list((((set(old_label) & set(current_label)) | set(prompt_label_list)) - set(negative_duplicate_list)))
        else:
            yet_another_unique_label = []
            negative_duplicate_list = []

        # add the corrected mask to predictor
        for object_id, object_mask in per_obj_input_mask.items():
            if idx == 0 or (idx > 0 and object_id in yet_another_unique_label and object_id not in negative_duplicate_list):
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx,
                    obj_id=object_id,
                    mask=object_mask,
                )
        
        if len(inference_state['point_inputs_per_obj']) == 0 and len(inference_state['mask_inputs_per_obj']) == 0:
            print("Empty overseer mask, using dummy background point prompt. Happens when camera move out of scene.")
            dummy_id = num_classes if shift_by_1 else 0
            dummy_mask = np.zeros((height, width), dtype=bool)
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=idx,
                obj_id=dummy_id,
                mask=dummy_mask,
            )

        trigger_next_idx = None
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state): 
            idx = out_frame_idx
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            current_overseer_mask = ensure_cached_overseer_prediction(
                cache=per_obj_input_mask_cached,
                frame_idx=out_frame_idx,
                frame_names=frame_names,
                frame_path_by_name=frame_path_by_name,
                overseer_model=overseer_model,
                reshape_size=(height, width),
            )
            disagreement_metrics = compute_disagreement_metrics(
                per_obj_sam_mask=per_obj_output_mask,
                per_obj_overseer_mask=current_overseer_mask,
                height=height,
                width=width,
            )
            boundary_distance = None
            if enable_boundary_distance_gate:
                boundary_distance = compute_boundary_distance(
                    disagreement_metrics["sam_foreground"],
                    disagreement_metrics["overseer_foreground"],
                )

            if confidence_dir is not None and len(out_obj_ids) > 0:
                confidence_map = compute_confidence_map(
                    out_mask_logits,
                    object_ids=out_obj_ids,
                    score_thresh=score_thresh,
                )
                confidence_path = os.path.join(
                    confidence_dir,
                    f"{frame_names[out_frame_idx]}_confidence.png",
                )
                save_confidence_map(confidence_path, confidence_map)
                confidence_stats = summarise_confidence_map(confidence_map)
                inference_rows[frame_names[out_frame_idx]] = {
                    "video_name": video_name,
                    "frame_name": frame_names[out_frame_idx],
                    "frame_idx": out_frame_idx,
                    "segment_id": segment_id,
                    "num_objects": len(out_obj_ids),
                    "confidence_path": confidence_path,
                    **confidence_stats,
                }
            else:
                inference_rows[frame_names[out_frame_idx]] = {
                    "video_name": video_name,
                    "frame_name": frame_names[out_frame_idx],
                    "frame_idx": out_frame_idx,
                    "segment_id": segment_id,
                    "num_objects": len(out_obj_ids),
                    "confidence_path": None,
                    "confidence_mean": None,
                    "confidence_std": None,
                    "confidence_min": None,
                    "confidence_max": None,
                }
            # write the output masks as palette PNG files to output_mask_dir
            save_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                output_palette=palette,
                save_binary_mask=save_binary_mask,
                num_classes=num_classes,
                shift_by_1=shift_by_1,
            )

            if overseer_mask_dir is not None:
                save_masks_to_dir(
                    output_mask_dir=overseer_mask_dir,
                    video_name=video_name,
                    frame_name=frame_names[out_frame_idx],
                    per_obj_output_mask=current_overseer_mask,
                    height=height,
                    width=width,
                    output_palette=palette,
                    save_binary_mask=save_binary_mask,
                    num_classes=num_classes,
                    shift_by_1=shift_by_1,
                )

            gt_mask_path = resolve_gt_mask_path(
                frame_name=frame_names[out_frame_idx],
                video_dir=video_dir,
                video_name=video_name,
                base_video_dir=base_video_dir,
                gt_root_dir=gt_root_dir,
            )
            gt_metrics = {
                "gt_mask_path": gt_mask_path,
                "gt_pixel_accuracy": None,
                "gt_macro_iou": None,
                "gt_macro_dice": None,
                "gt_error_rate": None,
            }
            if gt_mask_path is not None:
                pred_label_map = per_obj_mask_to_label_map(
                    per_obj_mask=per_obj_output_mask,
                    height=height,
                    width=width,
                    shift_by_1=shift_by_1,
                )
                gt_label_map = load_dataset_mask(gt_mask_path, gt_dataset_config)
                gt_frame_metrics = compute_frame_metrics(
                    pred_mask=pred_label_map,
                    gt_mask=gt_label_map,
                    dataset_config=gt_dataset_config,
                )
                if gt_frame_metrics["macro_iou"] is not None:
                    gt_metrics = {
                        "gt_mask_path": gt_mask_path,
                        "gt_pixel_accuracy": gt_frame_metrics["pixel_accuracy"],
                        "gt_macro_iou": gt_frame_metrics["macro_iou"],
                        "gt_macro_dice": gt_frame_metrics["macro_dice"],
                        "gt_error_rate": gt_frame_metrics["error_rate"],
                    }
                    gt_macro_ious.append(gt_frame_metrics["macro_iou"])
                    gt_macro_dices.append(gt_frame_metrics["macro_dice"])
                    gt_pixel_accuracies.append(gt_frame_metrics["pixel_accuracy"])
                    video_summary["gt_frames_evaluated"] += 1

            bad_disagreement_frame = False
            if enable_disagreement_gate and out_frame_idx > segment_start_idx:
                bad_disagreement_frame = disagreement_metrics["iou"] < disagreement_iou_threshold
                if enable_boundary_distance_gate and boundary_distance is not None:
                    bad_disagreement_frame = (
                        bad_disagreement_frame
                        or boundary_distance > boundary_distance_threshold
                    )
                disagreement_counter = disagreement_counter + 1 if bad_disagreement_frame else 0
            else:
                disagreement_counter = 0

            video_summary["frames_processed"] += 1
            per_frame_ious.append(disagreement_metrics["iou"])

            future_n_frame = min(10, len(frame_names) - idx)
            per_obj_input_mask_n = []
            duplicate_list = []
            negative_duplicate_list = []
            partial_duplicate_list = []
            prompt_label_list = []
            class_change_trigger = False
            disagreement_trigger = False
            reprompt_executed = False
            reprompt_reason = ""

            # getting overseer prediction and caching them for future use
            for n in range(future_n_frame):
                per_obj_input_mask_n.append(
                    ensure_cached_overseer_prediction(
                        cache=per_obj_input_mask_cached,
                        frame_idx=idx + n,
                        frame_names=frame_names,
                        frame_path_by_name=frame_path_by_name,
                        overseer_model=overseer_model,
                        reshape_size=(height, width),
                    )
                )

            # getting separate unique labels for each frame in per_obj_input_mask_n
            unique_lable_n = get_unique_label(per_obj_input_mask_n)
            if idx == 0:
                old_label = unique_lable_n[0]
            else:
                current_label = unique_lable_n[0]
                
                # to restart the inference after 50 frames to avoid false labels to continue longer
                buffer_length -= 1                    
                if buffer_length == 0:
                    trigger_next_idx = min(idx + 1, len(frame_names))
                    break
                
                if old_label != current_label:
                    # only if current label have new objects compared to old label (when tool entering scene). But not when a tool exiting!
                    #TODO: Can define this as parameter and be more dynamic
                    # also ignoring labels that appear in one frame but not in next 3 frames
                    new_obj_label = list(set(current_label) - set(old_label))
                    unique_lable_n = unique_lable_n[1:4]
                    for unique_l in unique_lable_n:
                        new_obj_label = list(set(new_obj_label) & set(unique_l))
                    
                    if new_obj_label:
                        # find out exact duplicates of new_obj_label, and run analysis on them
                        for new_obj in new_obj_label:
                            mask1 = per_obj_input_mask_n[0][new_obj]
                            for obj in current_label:
                                mask2 = per_obj_input_mask_n[0][obj]

                                true_positions_1 = (mask1 == 1)
                                true_positions_2 = (mask2 == 1)
                                matching_true_positions = np.logical_and(true_positions_1, true_positions_2)
                                similarity_1 = np.sum(matching_true_positions) / np.sum(true_positions_1)
                                similarity_2 = np.sum(matching_true_positions) / np.sum(true_positions_2)

                                # true duplicate
                                if similarity_1 >= 0.70 and similarity_2 >= 0.70 and obj != new_obj:
                                    # directly append if empty, else check if the reverse pair already exist
                                    if not duplicate_list:
                                        duplicate_list.append([obj, new_obj])
                                    else:
                                        common_element_flag = False
                                        for sublist in duplicate_list:
                                            if list(set(sublist) & set([obj, new_obj])):
                                                common_element_flag = True
                                                new_element = list(set([obj, new_obj]) - set(sublist))
                                                if new_element: sublist.append(new_element[0])
                                        if not common_element_flag:
                                            duplicate_list.append([obj, new_obj])
                                
                                # if one similarity is high but not other way, partial duplicate and remove the smaller label
                                elif similarity_1 >= 0.70 and similarity_2 < 0.70:
                                    partial_duplicate_list.append(new_obj)
                        
                        # if new duplicate is being replaced, we need to remove it (Kinda doing this with false_positions, but need to be more explicit)                            
                        if duplicate_list:
                            negative_duplicate_list = duplicate_list
                            for sublist in duplicate_list:
                                # prompt label is the one that will actually be used for new prompting
                                # negative_duplicate_list just filters added sublist                           
                                selections, use_nnunet = choose_duplicate_label(per_obj_mask_n=per_obj_input_mask_n, duplicate_label=sublist)
                                
                                prompt_label = max(selections, key=selections.get)
                                # use nnunet almost in ensembling fashion for error propagation
                                if use_nnunet and nnunet_model is not None:
                                    per_obj_input_mask_nnunet = nnunet_model.get_prediction(frame_path_by_name[frame_names[idx]], reshape_size=(width, height))
                                    area_to_check = per_obj_input_mask_n[0][prompt_label]

                                    for key in selections:
                                        if key in per_obj_input_mask_nnunet:
                                            selections[key] += np.sum(np.logical_and(area_to_check, per_obj_input_mask_nnunet[key]))
                                    prompt_label = max(selections, key=selections.get)

                                if prompt_label not in old_label:
                                    prompt_label_list.append(prompt_label)
                                negative_duplicate_list = [[x for x in slist if x != prompt_label] for slist in negative_duplicate_list]

                        # if new label without any duplicate        
                        if new_obj_label:
                            single_label = list((set(new_obj_label) - set([x for xs in duplicate_list for x in xs])) - set(partial_duplicate_list))
                            if single_label:
                                for ixn in single_label:
                                    if ixn not in old_label:
                                        prompt_label_list.append(ixn)

                        if prompt_label_list:
                            if break_endless_loop:
                                break_endless_loop = False 
                                idx += 1
                                continue
                            class_change_trigger = True
                            reprompt_executed = True
                            reprompt_reason = "class-change"
                            video_summary["class_change_reprompts"] += 1
                            trigger_next_idx = idx
                            print(
                                f"[reprompt] video={video_name} frame={frame_names[idx]} idx={idx} "
                                f"reason=class-change labels={prompt_label_list}"
                            )
                        break_endless_loop = False

                old_label = list(set(old_label) & set(current_label)) + prompt_label_list

            if (
                not class_change_trigger
                and enable_disagreement_gate
                and out_frame_idx > segment_start_idx
                and disagreement_counter >= disagreement_bad_frames
            ):
                disagreement_trigger = True
                reprompt_executed = True
                reprompt_reason = "disagreement"
                trigger_next_idx = idx
                prompt_label_list = []
                negative_duplicate_list = []
                current_label = sorted(set(current_overseer_mask.keys()))
                old_label = current_label.copy()
                video_summary["disagreement_reprompts"] += 1
                print(
                    f"[reprompt] video={video_name} frame={frame_names[idx]} idx={idx} "
                    f"reason=disagreement iou={disagreement_metrics['iou']:.4f} "
                    f"counter={disagreement_counter}"
                )

            trace_row = {
                **inference_rows[frame_names[out_frame_idx]],
                "sam2_vs_overseer_iou": disagreement_metrics["iou"],
                "sam2_vs_overseer_fg_iou": disagreement_metrics["foreground_iou"],
                "sam2_vs_overseer_boundary_distance": boundary_distance,
                "disagreement_bad_frame": bad_disagreement_frame,
                "disagreement_counter": disagreement_counter,
                "class_change_trigger": class_change_trigger,
                "disagreement_trigger": disagreement_trigger,
                "reprompt_executed": reprompt_executed,
                "reprompt_reason": reprompt_reason,
                **gt_metrics,
            }
            trace_rows.append(trace_row)
            print(
                f"[trace] video={video_name} frame={frame_names[out_frame_idx]} idx={out_frame_idx} "
                f"segment={segment_id} iou={disagreement_metrics['iou']:.4f} "
                f"fg_iou={disagreement_metrics['foreground_iou']:.4f} "
                f"boundary={'n/a' if boundary_distance is None or not np.isfinite(boundary_distance) else f'{boundary_distance:.4f}'} "
                f"class_change={class_change_trigger} disagreement={disagreement_trigger} "
                f"bad={bad_disagreement_frame} counter={disagreement_counter} "
                f"reprompt={reprompt_executed} reason={reprompt_reason or 'none'}"
            )

            if (
                disagreement_visual_dir is not None
                and disagreement_visual_count < max_disagreement_visuals
                and (bad_disagreement_frame or reprompt_executed)
            ):
                visual_path = os.path.join(
                    disagreement_visual_dir,
                    f"{frame_names[out_frame_idx]}_segment{segment_id}_disagreement.png",
                )
                save_disagreement_visual(
                    path=visual_path,
                    frame_path=frame_path_by_name[frame_names[out_frame_idx]],
                    sam_foreground=disagreement_metrics["sam_foreground"],
                    overseer_foreground=disagreement_metrics["overseer_foreground"],
                )
                disagreement_visual_count += 1

            if reprompt_executed:
                break
            idx += 1
        if trigger_next_idx is None:
            if idx >= len(frame_names) - 1:
                idx = len(frame_names)
            else:
                idx += 1
        else:
            idx = trigger_next_idx
    if temp_subset_root is not None:
        try:
            for frame_info in frame_infos:
                os.unlink(os.path.join(predictor_video_dir, os.path.basename(frame_info["path"])))
            os.rmdir(predictor_video_dir)
            os.rmdir(temp_subset_root)
        except OSError:
            pass
    if per_frame_ious:
        video_summary["mean_iou"] = float(np.mean(per_frame_ious))
        video_summary["min_iou"] = float(np.min(per_frame_ious))
    if gt_macro_ious:
        video_summary["gt_macro_iou_mean"] = float(np.mean(gt_macro_ious))
        video_summary["gt_macro_dice_mean"] = float(np.mean(gt_macro_dices))
        video_summary["gt_pixel_accuracy_mean"] = float(np.mean(gt_pixel_accuracies))
    return trace_rows, video_summary


def main():
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
        choices=['MaskRCNN', 'DETR', 'Mask2Former'],

        help="path to the Overseer model checkpoint",
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
        choices=['CADIS', 'CHOLECSEG8K', 'CATARACT1K'],
        help="dataset type to run the prediction on. Currently supported: CADIS, CHOLECSEG8K, CATARACT1K",
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
        "--video_name",
        type=str,
        help="optional single video/clip folder name to run, e.g. video01_00080",
    )
    parser.add_argument(
        "--video_names",
        type=str,
        nargs="+",
        help="optional list of video/clip folder names to run, e.g. video01_00080 video01_00160",
    )
    parser.add_argument(
        "--frame_name",
        type=str,
        help="optional single frame stem to run, e.g. frame_80_endo or 0080",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        help="optional first frame index to process",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        help="optional last frame index to process",
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
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks"
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
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
        "--gt_root_dir",
        type=str,
        help="optional root directory for ground-truth masks. If omitted, GT masks are searched next to the input frames.",
    )
    parser.add_argument(
        "--enable_disagreement_gate",
        action="store_true",
        help="enable disagreement-gated corrective re-prompting.",
    )
    parser.add_argument(
        "--disagreement_iou_threshold",
        type=float,
        default=0.5,
        help="trigger a bad disagreement frame when SAM2 vs Overseer IoU falls below this value.",
    )
    parser.add_argument(
        "--disagreement_bad_frames",
        type=int,
        default=2,
        help="minimum consecutive bad disagreement frames before corrective re-prompting.",
    )
    parser.add_argument(
        "--enable_boundary_distance_gate",
        action="store_true",
        help="also allow disagreement triggering using foreground boundary distance.",
    )
    parser.add_argument(
        "--boundary_distance_threshold",
        type=float,
        default=20.0,
        help="boundary-distance trigger threshold in pixels when boundary gating is enabled.",
    )
    parser.add_argument(
        "--save_disagreement_visuals",
        action="store_true",
        help="save lightweight disagreement overlay images for bad/triggered frames.",
    )
    parser.add_argument(
        "--max_disagreement_visuals",
        type=int,
        default=10,
        help="maximum number of disagreement debug visuals to save per video.",
    )
    args = parser.parse_args()
    if args.frame_name is not None and (args.start_frame is not None or args.end_frame is not None):
        raise ValueError("--frame_name cannot be combined with --start_frame/--end_frame")
    if args.disagreement_bad_frames < 1:
        raise ValueError("--disagreement_bad_frames must be >= 1")
    if args.video_name is not None and args.video_names is not None:
        raise ValueError("Use either --video_name or --video_names, not both")
    if str(args.device).lower().startswith("cpu"):
        configure_cpu_runtime()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )
    video_entries = discover_video_dirs(args.base_video_dir)
    if not video_entries:
        raise RuntimeError(f"No video folders with frames found under {args.base_video_dir}")

    requested_video_names = None
    if args.video_name is not None:
        requested_video_names = {args.video_name}
    elif args.video_names is not None:
        requested_video_names = set(args.video_names)

    if requested_video_names is not None:
        video_entries = [entry for entry in video_entries if entry[0] in requested_video_names]
        if not video_entries:
            raise RuntimeError(f"Requested video(s) not found under {args.base_video_dir}: {sorted(requested_video_names)}")
        found_names = {name for name, _ in video_entries}
        missing_names = sorted(requested_video_names - found_names)
        if missing_names:
            raise RuntimeError(f"Requested video(s) not found under {args.base_video_dir}: {missing_names}")

    video_names = [name for name, _ in video_entries]

    # adding this filter based on the dataset type
    if args.dataset_type == "CADIS":
        # video_names[:] = sorted([item for item in video_names if item.startswith('train')])
        num_classes = 18
        ignore_indices = [255]
        shift_by_1 = True
        maskrcnn_hidden_ft = 32
        maskrcnn_backbone = 'ResNet18'
        
    elif args.dataset_type == "CHOLECSEG8K":
        # video_names[:] = sorted([item for item in video_names if item.startswith('video')])
        num_classes = 13
        ignore_indices = [0] if args.overseer_type == "Mask2Former" else []
        shift_by_1 = False
        maskrcnn_hidden_ft = 64
        maskrcnn_backbone = 'ResNet50'

    elif args.dataset_type == "CATARACT1K":
        # video_names[:] = sorted([item for item in video_names if item.startswith('case')])
        num_classes = 14
        ignore_indices = [0] if args.overseer_type == "DETR" or args.overseer_type == "Mask2Former" else []
        shift_by_1 = False
        maskrcnn_hidden_ft = 32
        maskrcnn_backbone = 'ResNet18'

    else:        
        raise NotImplementedError

    palette = get_primary_visual_palette(num_classes)
    gt_dataset_config = get_dataset_config(args.dataset_type)

    if args.overseer_type == "MaskRCNN":
        maskrcnn_model = get_model_instance_segmentation(
            num_classes=num_classes,
            trainable_backbone_layers=0,
            hidden_ft=maskrcnn_hidden_ft,
            custom_in_ft_box=None,
            custom_in_ft_mask=None,
            backbone=maskrcnn_backbone,
            img_size=(299, 299)
        )
        maskrcnn_model.load_state_dict(torch.load(args.overseer_checkpoint, weights_only=True, map_location='cpu'))
        maskrcnn_model = maskrcnn_model.to(args.device)
        maskrcnn_model.eval()       
        overseer_model = MaskRCNN(maskrcnn_model, shift_by_1, ignore_indices, num_classes, device=args.device)

    elif args.overseer_type == "DETR":
        num_train_classes = num_classes - len(ignore_indices) + 1
        detr_model = DetrForSegmentation.from_pretrained(
            "facebook/detr-resnet-50-panoptic",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=100).to(args.device)
        detr_model.load_state_dict(torch.load(args.overseer_checkpoint, map_location='cpu', weights_only=True))
        detr_model.eval()
        overseer_model = DETR(detr_model, shift_by_1, ignore_indices, num_classes, num_train_classes, args.dataset_type, device=args.device)
    
    elif args.overseer_type == "Mask2Former":
        num_train_classes = num_classes - len(ignore_indices) + 1
        mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-instance",
            num_labels=num_train_classes,
            ignore_mismatched_sizes=True,
            num_queries=20).to(args.device)
        mask2former_model.load_state_dict(torch.load(args.overseer_checkpoint, map_location='cpu', weights_only=True))
        mask2former_model.eval()
        overseer_model = Mask2Former(mask2former_model, shift_by_1, ignore_indices, num_classes, num_train_classes, args.dataset_type, device=args.device)

    else: 
        raise NotImplementedError
    
    if args.nnunet_checkpoint is not None:
        nnunet_predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(args.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        nnunet_predictor.initialize_from_trained_model_folder(
            args.nnunet_checkpoint,
            use_folds=(0,),
            checkpoint_name='checkpoint_final.pth',
        )
        nnunet_model = nnUNet(nnunet_predictor, ignore_indices)
    else: nnunet_model = None

    print(f"running SASVI prediction on {len(video_names)} videos:\n{video_names}")
    start_time = time.time()
    inference_rows = []
    video_summaries = []
    
    for n_video, (video_name, video_dir) in enumerate(video_entries):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        video_rows, video_summary = sasvi_inference(
            predictor=predictor,
            base_video_dir=args.base_video_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            video_dir=video_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_name=args.frame_name,

            overseer_type=args.overseer_type,
            overseer_mask_dir=args.overseer_mask_dir,
            overseer_model=overseer_model,
            nnunet_model=nnunet_model,

            num_classes=num_classes,
            ignore_indices=ignore_indices,
            shift_by_1=shift_by_1,
            palette=palette,
            dataset_type=args.dataset_type,
            gt_dataset_config=gt_dataset_config,
            gt_root_dir=args.gt_root_dir,
            
            score_thresh=args.score_thresh,
            save_binary_mask=args.save_binary_mask,
            analysis_output_dir=args.analysis_output_dir,
            enable_disagreement_gate=args.enable_disagreement_gate,
            disagreement_iou_threshold=args.disagreement_iou_threshold,
            disagreement_bad_frames=args.disagreement_bad_frames,
            enable_boundary_distance_gate=args.enable_boundary_distance_gate,
            boundary_distance_threshold=args.boundary_distance_threshold,
            save_disagreement_visuals=args.save_disagreement_visuals,
            max_disagreement_visuals=args.max_disagreement_visuals,
        )
        inference_rows.extend(video_rows)
        video_summaries.append(video_summary)

    if args.analysis_output_dir is not None and len(inference_rows) > 0:
        metadata_path = os.path.join(args.analysis_output_dir, "inference", "inference_metadata.csv")
        write_rows_to_csv(metadata_path, inference_rows, INFERENCE_METADATA_COLUMNS)
        report_path = os.path.join(args.analysis_output_dir, "inference", "disagreement_gate_report.md")
        write_markdown_report(
            report_path,
            config={
                "device": args.device,
                "dataset_type": args.dataset_type,
                "base_video_dir": args.base_video_dir,
                "gt_root_dir": args.gt_root_dir,
                "video_name": args.video_name,
                "video_names": args.video_names,
                "frame_name": args.frame_name,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "enable_disagreement_gate": args.enable_disagreement_gate,
                "disagreement_iou_threshold": args.disagreement_iou_threshold,
                "disagreement_bad_frames": args.disagreement_bad_frames,
                "enable_boundary_distance_gate": args.enable_boundary_distance_gate,
                "boundary_distance_threshold": args.boundary_distance_threshold,
                "save_disagreement_visuals": args.save_disagreement_visuals,
                "max_disagreement_visuals": args.max_disagreement_visuals,
            },
            video_summaries=video_summaries,
            trace_rows=inference_rows,
        )

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.6f} seconds")
    print(f"completed SASVI prediction on {len(video_names)} videos -- "
          f"output masks saved to {args.output_mask_dir}"    
    )

if __name__ == "__main__":
    main()
