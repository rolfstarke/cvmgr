import logging
logger = logging.getLogger('cvmgr')
import fiftyone 
import torch
import fiftyone.brain
import fiftyone.utils.labels
from .sam3_get_model import sam3_get_model
from .mask_to_polyline import mask_to_polyline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def sam3_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False):
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
    samples_without_masks = dataset.filter_labels(
        "ground_truth", 
        fiftyone.ViewField("mask").is_null()
    )
    logger.info(f"Remaining samples without masks: {len(samples_without_masks)}")

    return dataset