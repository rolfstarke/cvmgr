import pathlib
import fiftyone
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from .logging_check import util_log

# Why did the SAHI model get arrested? For slicing up the competition into tiny, overlapping pieces.

@util_log(
    "evaluate_model_sahi",
    success_text=lambda result, args, kwargs: (
        f"ran {result} models → {kwargs.get('prediction_labelfield', 'yolo_predictions_sahi')} "
        f"on {kwargs.get('dataset_name', 'mark_lane_leuthener_21_ORIGINAL')}"
    ),
)
def evaluate_model_sahi(
    dataset_name: str = "mark_lane_leuthener_21_ORIGINAL",
    prediction_labelfield: str = "yolo_predictions_sahi",
    replace: bool = False,
    model_path: str | None = None,
    slice_height: int = 320,
    slice_width: int = 320,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    confidence_threshold: float = 0.25,
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
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(mp),
            confidence_threshold=confidence_threshold,
        )
        for sample in dataset.iter_samples(progress=True, autosave=True):
            result = get_sliced_prediction(
                sample.filepath,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                verbose=0,
            )
            sample[prediction_labelfield] = fiftyone.Detections(
                detections=result.to_fiftyone_detections()
            )

    dataset.save()
    return len(model_paths)
