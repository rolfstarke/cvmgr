import pathlib
import logging
logger = logging.getLogger('cvmgr')
import os
import fiftyone
from ultralytics import YOLO

def evaluate_model(prediction_labelfield: str, prediction_model: str = None, conf: float = 0.3, replace: bool = False):

    dataset = fiftyone.load_dataset("mark_lane_leuthener_custom_21_hd")

    if replace:
        if prediction_labelfield in dataset.get_field_schema():
            dataset.delete_sample_field(prediction_labelfield)
    if prediction_model:
        model = YOLO(prediction_model)
        if prediction_labelfield in dataset.get_field_schema():
            dataset.delete_sample_field(prediction_labelfield)
        dataset.apply_model(model, label_field=prediction_labelfield, confidence_thresh=conf)

    if prediction_labelfield in dataset.get_field_schema():
        dataset.evaluate_detections(
                prediction_labelfield,
                gt_field="ground_truth_masks",
                eval_key=f"{prediction_labelfield}_eval",
                compute_mAP=True
            )
    
    dataset.save()

