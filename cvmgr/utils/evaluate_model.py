import pathlib
import fiftyone
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from .logging_check import util_log

# Why did the model fail peer review? Because it kept predicting "no issues" on its own weights.

# Heater models were trained before the label was standardised to "Radiator" in the GT.
_LABEL_MAP = {"Heater": "Radiator"}

# GT uses "Conference chair" and "Glass internal"; models trained on the broader labels below.
# Keep both so COCO classwise matching credits either.
_LABEL_COPIES = [("Desk chair", "Conference chair"), ("Glass external", "Glass internal")]

@util_log(
    "evaluate_model",
    success_text=lambda result, args, kwargs: f"ran {result} models → {kwargs.get('prediction_labelfield', 'yolo_predictions_' + kwargs.get('source', 'ray'))} on {kwargs.get('dataset_name', 'mark_lane_leuthener_21_ORIGINAL')}",
)
def evaluate_model(
    dataset_name: str = "mark_lane_leuthener_21_ORIGINAL",
    prediction_labelfield: str | None = None,
    replace: bool = False,
    model_path: str | None = None,
    source: str = "ray",
):
    if prediction_labelfield is None:
        prediction_labelfield = f"yolo_predictions_{source}"

    dataset = fiftyone.load_dataset(dataset_name)

    if replace and prediction_labelfield in dataset.get_field_schema():
        dataset.delete_sample_field(prediction_labelfield)

    if model_path is not None:
        model_paths = [pathlib.Path(model_path)]
    else:
        model_paths = sorted((pathlib.Path(__file__).parents[2] / "models" / source).glob("*.pt"))

    for mp in model_paths:
        model = YOLO(str(mp))
        raw_classes = set(model.names.values())
        mapped_classes = {_LABEL_MAP.get(c, c) for c in raw_classes}
        copy_dsts = {dst for src, dst in _LABEL_COPIES if src in mapped_classes}
        classes_to_replace = raw_classes | mapped_classes | copy_dsts

        for sample in dataset.iter_samples(progress=True, autosave=True):
            results = model(sample.filepath, verbose=False)
            new_dets = []
            for r in results:
                h, w = r.orig_shape
                has_masks = r.masks is not None and len(r.masks) > 0
                if has_masks:
                    h_m, w_m = r.masks.data.shape[1], r.masks.data.shape[2]
                    sy, sx = h_m / h, w_m / w

                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_label = model.names[int(box.cls)]

                    mask = None
                    if has_masks:
                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        my1 = max(0, int(iy1 * sy))
                        my2 = min(h_m, int(iy2 * sy) + 1)
                        mx1 = max(0, int(ix1 * sx))
                        mx2 = min(w_m, int(ix2 * sx) + 1)
                        crop = r.masks.data[i, my1:my2, mx1:mx2].float()
                        bh, bw = max(1, iy2 - iy1), max(1, ix2 - ix1)
                        mask = (
                            F.interpolate(
                                crop.unsqueeze(0).unsqueeze(0),
                                size=(bh, bw),
                                mode="bilinear",
                                align_corners=False,
                            )[0, 0].cpu().numpy() > 0.5
                        )

                    label = _LABEL_MAP.get(raw_label, raw_label)
                    new_dets.append(fiftyone.Detection(
                        label=label,
                        bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                        mask=mask,
                        confidence=float(box.conf),
                    ))
                    for src, dst in _LABEL_COPIES:
                        if label == src:
                            new_dets.append(fiftyone.Detection(
                                label=dst,
                                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                                mask=mask.copy() if mask is not None else None,
                                confidence=float(box.conf),
                            ))

            try:
                existing_field = sample.get_field(prediction_labelfield)
                existing = existing_field.detections if existing_field else []
            except AttributeError:
                existing = []
            kept = [d for d in existing if d.label not in classes_to_replace]
            sample[prediction_labelfield] = fiftyone.Detections(detections=kept + new_dets)

    dataset.save()
    return len(model_paths)
