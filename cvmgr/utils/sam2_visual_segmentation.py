import logging
import os
import pathlib

import fiftyone
import yaml

logger = logging.getLogger('cvmgr')


def sam2_visual_segmentation(dataset: fiftyone.core.dataset.Dataset = None, recalculate: bool = False):
    if dataset is None:
        return dataset

    with (pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml").open("r") as file:
        resources = yaml.safe_load(file)
    os.environ["CUDA_VISIBLE_DEVICES"] = resources["sam"]["cuda_visible_devices"]

    samples_without_masks = dataset.filter_labels("ground_truth", fiftyone.ViewField("mask").is_null())
    if len(samples_without_masks) == 0:
        return dataset

    logger.info(f"Adding masks to {len(samples_without_masks)} samples")
    model = fiftyone.zoo.load_zoo_model("segment-anything-2-hiera-large-image-torch")
    samples_without_masks.apply_model(model, label_field="ground_truth", prompt_field="ground_truth")
    return dataset