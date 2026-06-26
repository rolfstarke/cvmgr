import os

import fiftyone
from .sam3_get_model import sam3_get_model
from .logging_check import util_log


@util_log("sam3_visual_segmentation", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "missing_masks == 0 OR no_work")
def sam3_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False, mask_threshold: float = 0.5, gpu: str = "0"):
    if dataset is None:
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if recalculate:
        for sample in dataset.iter_samples(autosave=True):
            if sample.ground_truth:
                for detection in sample.ground_truth.detections:
                    detection.mask = None

    samples_needing_masks = dataset.filter_labels(
        "ground_truth",
        fiftyone.ViewField("mask").is_null()
    )
    if len(samples_needing_masks) == 0:
        return dataset

    fiftyone.zoo.register_zoo_model_source("https://github.com/harpreetsahota204/sam3_images")
    sam3_get_model()

    model = fiftyone.zoo.load_zoo_model("facebook/sam3")
    model.operation = "visual_segmentation"
    model.mask_threshold = mask_threshold

    samples_needing_masks.apply_model(
        model,
        label_field="ground_truth",
        prompt_field="ground_truth",
        batch_size=16,
        num_workers=8,
    )

    remaining = dataset.filter_labels("ground_truth", fiftyone.ViewField("mask").is_null())
    return False if len(remaining) > 0 else dataset