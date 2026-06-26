import pathlib
import fiftyone
from ultralytics import YOLO
from .logging_check import util_log

# Why did the model fail peer review? Because it kept predicting "no issues" on its own weights.

@util_log(
    "evaluate_model",
    success_text=lambda result, args, kwargs: f"ran {result} models → {kwargs.get('prediction_labelfield', 'yolo_predictions')} on {kwargs.get('dataset_name', 'mark_lane_leuthener_21_ORIGINAL')}",
)
def evaluate_model(
    dataset_name: str = "mark_lane_leuthener_21_ORIGINAL",
    prediction_labelfield: str = "yolo_predictions",
    replace: bool = False,
    model_path: str | None = None,
):
    dataset = fiftyone.load_dataset(dataset_name)

    if replace and prediction_labelfield in dataset.get_field_schema():
        dataset.delete_sample_field(prediction_labelfield)

    if model_path is not None:
        model_paths = [pathlib.Path(model_path)]
    else:
        models_dir = pathlib.Path(__file__).parents[2] / "models"
        model_paths = sorted(models_dir.glob("*.pt"))

    for mp in model_paths:
        model = YOLO(str(mp))
        dataset.apply_model(model, label_field=prediction_labelfield)

    dataset.save()
    return len(model_paths)
