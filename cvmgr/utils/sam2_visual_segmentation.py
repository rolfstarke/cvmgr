import os

import fiftyone
from .logging_check import util_log

@util_log("sam2_visual_segmentation", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "missing_masks == 0 OR no_work")
def sam2_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False, gpu: str = "0"):
    if dataset is None:
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    samples_without_masks = dataset.filter_labels("ground_truth", fiftyone.ViewField("mask").is_null())
    if len(samples_without_masks) == 0:
        return dataset

    model = fiftyone.zoo.load_zoo_model("segment-anything-2-hiera-large-image-torch")
    samples_without_masks.apply_model(model, label_field="ground_truth", prompt_field="ground_truth")
    remaining = dataset.filter_labels("ground_truth", fiftyone.ViewField("mask").is_null())
    if len(remaining) > 0:
        return False
    return dataset