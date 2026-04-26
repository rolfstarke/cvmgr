import logging
logger = logging.getLogger('cvmgr')
import fiftyone 
import torch
from .sam3_get_model import sam3_get_model
from .mask_to_polyline import mask_to_polyline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def sam2_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False):
    device = torch.cuda.is_available()
    logger.info(f"CUDA available: {device}")

    # Filter to only samples that have ground_truth detections but no masks
    samples_without_masks = dataset.filter_labels(
        "ground_truth", 
        fiftyone.ViewField("mask").is_null()
    )
    if len(samples_without_masks) == 0:
        logger.info("All ground truth samples already have masks")
        return dataset
    logger.info(f"Adding masks to {len(samples_without_masks)} samples")

    model = fiftyone.zoo.load_zoo_model("segment-anything-2-hiera-large-image-torch") 
    samples_without_masks.apply_model(model, label_field="ground_truth", prompt_field="ground_truth")
    samples_without_masks = dataset.filter_labels(
        "ground_truth", 
        fiftyone.ViewField("mask").is_null()
    )
    logger.info(f"Remaining samples without masks: {len(samples_without_masks)}")
    
    mask_to_polyline(dataset_name=dataset.name, mask_field="ground_truth")

    return dataset