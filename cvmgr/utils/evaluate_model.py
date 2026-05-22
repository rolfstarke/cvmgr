import fiftyone
from ultralytics import YOLO
from .logging_check import util_log

@util_log("evaluate_model", success_text=lambda result, args, kwargs: "prediction_field_present AND evaluation_saved")
def evaluate_model(
    prediction_model: str,
    prediction_labelfield: str,
    conf: float = 0.3,
    gt_field: str = "ground_truth",
):
    dataset = fiftyone.load_dataset("mark_lane_leuthener_custom_21_lowres")

    # Always (re-)run predictions and overwrite existing field
    if prediction_labelfield in dataset.get_field_schema():
        dataset.delete_sample_field(prediction_labelfield)

    model = YOLO(prediction_model)
    dataset.apply_model(model, label_field=prediction_labelfield, confidence_thresh=conf)

    dataset.evaluate_detections(
        prediction_labelfield,
        gt_field=gt_field,
        eval_key=f"{prediction_labelfield}_eval",
        compute_mAP=True,
    )

    dataset.save()
    return True

