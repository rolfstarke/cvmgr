import pathlib
import fiftyone
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from .logging_check import util_log

# Why did the SAHI model get arrested? For slicing up the competition into tiny, overlapping pieces.

# Heater models were trained before the label was standardised to "Radiator" in the GT.
_LABEL_MAP = {"Heater": "Radiator"}

# GT uses "Conference chair" and "Glass internal"; models trained on the broader labels below.
# Keep both so COCO classwise matching credits either.
_LABEL_COPIES = [("Desk chair", "Conference chair"), ("Glass external", "Glass internal")]

@util_log(
    "evaluate_model_sahi",
    success_text=lambda result, args, kwargs: (
        f"ran {result} models → {kwargs.get('prediction_labelfield') or 'yolo_predictions_sahi_' + kwargs.get('source', 'ray')} "
        f"on {kwargs.get('dataset_name', 'mark_lane_leuthener_21_ORIGINAL')}"
    ),
)
def evaluate_model_sahi(
    dataset_name: str = "mark_lane_leuthener_21_ORIGINAL",
    prediction_labelfield: str | None = None,
    replace: bool = False,
    model_path: str | None = None,
    source: str = "ray",
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    confidence_threshold: float = 0.25,
    postprocess_match_threshold: float = 0.1,
):
    if prediction_labelfield is None:
        prediction_labelfield = f"yolo_predictions_sahi_{source}"

    dataset = fiftyone.load_dataset(dataset_name)

    if replace and prediction_labelfield in dataset.get_field_schema():
        dataset.delete_sample_field(prediction_labelfield)

    if model_path is not None:
        model_paths = [pathlib.Path(model_path)]
    else:
        model_paths = sorted((pathlib.Path(__file__).parents[2] / "models" / source).glob("*.pt"))

    for mp in model_paths:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(mp),
            confidence_threshold=confidence_threshold,
        )
        raw_classes = set(detection_model.category_mapping.values())
        mapped_classes = {_LABEL_MAP.get(c, c) for c in raw_classes}
        copy_dsts = {dst for src, dst in _LABEL_COPIES if src in mapped_classes}
        classes_to_replace = raw_classes | mapped_classes | copy_dsts

        for sample in dataset.iter_samples(progress=True, autosave=True):
            result = get_sliced_prediction(
                sample.filepath,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_match_metric="IOS",
                postprocess_match_threshold=postprocess_match_threshold,
                verbose=0,
            )
            try:
                existing_field = sample.get_field(prediction_labelfield)
                existing = existing_field.detections if existing_field else []
            except AttributeError:
                existing = []
            kept = [d for d in existing if d.label not in classes_to_replace]
            img_h, img_w = result.image_height, result.image_width
            new_dets = []
            for pred in result.object_prediction_list:
                x1, y1, x2, y2 = int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy)
                mask = None
                if pred.mask is not None and pred.mask.bool_mask is not None:
                    # .copy() so the small bbox crop doesn't hold a reference to
                    # the full-image bool_mask array (which can be GBs for large images)
                    mask = pred.mask.bool_mask[y1:y2, x1:x2].copy()
                label = _LABEL_MAP.get(pred.category.name, pred.category.name)
                new_dets.append(fiftyone.Detection(
                    label=label,
                    bounding_box=[x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h],
                    mask=mask,
                    confidence=float(pred.score.value),
                ))
                for src, dst in _LABEL_COPIES:
                    if label == src:
                        new_dets.append(fiftyone.Detection(
                            label=dst,
                            bounding_box=[x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h],
                            mask=mask.copy() if mask is not None else None,
                            confidence=float(pred.score.value),
                        ))
            del result  # free the full-image bool_masks before writing to FiftyOne
            sample[prediction_labelfield] = fiftyone.Detections(
                detections=kept + new_dets
            )

    dataset.save()
    return len(model_paths)
