import fiftyone 
import torch
import cv2
import numpy as np
from .sam3_get_model import sam3_get_model
import os
import pathlib
import yaml
from .logging_check import util_log

_resources_path = pathlib.Path(__file__).parent.parent / "configs" / "resources.yaml"


def _filter_mask_patches(mask: np.ndarray, keep: int) -> np.ndarray:
    """Keep only the `keep` largest connected components in a boolean mask."""
    uint8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(uint8, connectivity=8)
    if n <= keep + 1:  # +1 for background label 0
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
    top_labels = np.argsort(areas)[::-1][:keep] + 1  # +1 offset back to label ids
    clean = np.isin(labels, top_labels)
    return clean


@util_log("sam3_visual_segmentation", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "missing_masks == 0 OR no_work")
def sam3_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False, patchfilter: int = 4, sizefilter: float = 0.05):
    if dataset is None:
        return False

    with _resources_path.open("r") as f:
        resources = yaml.safe_load(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = resources.get("sam", {}).get("cuda_visible_devices", "0")

    if recalculate:
        for sample in dataset.iter_samples(autosave=True):
            dets = sample.ground_truth
            if not dets or not dets.detections:
                continue
            for det in dets.detections:
                det.mask = None

    # Filter to only samples that have ground_truth detections but no masks
    samples_without_masks = dataset.filter_labels(
        "ground_truth",
        fiftyone.ViewField("mask").is_null()
    )

    if len(samples_without_masks) == 0:
        return dataset
    
    fiftyone.zoo.register_zoo_model_source(
        "https://github.com/harpreetsahota204/sam3_images"
    )

    sam3_get_model()
    model = fiftyone.zoo.load_zoo_model("facebook/sam3")
    model.operation = "visual_segmentation"
    
    samples_without_masks.apply_model(
        model, 
        label_field="ground_truth",
        prompt_field="ground_truth", 
        batch_size=16, 
        num_workers=8
    )

    if (patchfilter and patchfilter > 0) or (sizefilter and sizefilter > 0):
        for sample in dataset.iter_samples(autosave=True):
            dets = sample.ground_truth
            if not dets or not dets.detections:
                continue

            if patchfilter and patchfilter > 0:
                for det in dets.detections:
                    if det.mask is not None:
                        det.mask = _filter_mask_patches(det.mask, patchfilter)

            if sizefilter and sizefilter > 0:
                areas = [int(det.mask.sum()) if det.mask is not None else 0 for det in dets.detections]
                max_area = max(areas) if areas else 0
                if max_area > 0:
                    threshold = sizefilter * max_area
                    dets.detections = [det for det, area in zip(dets.detections, areas) if area >= threshold]
                    sample.ground_truth = dets

    samples_without_masks = dataset.filter_labels(
        "ground_truth", 
        fiftyone.ViewField("mask").is_null()
    )
    if len(samples_without_masks) > 0:
        return False

    return dataset