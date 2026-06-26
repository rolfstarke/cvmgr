import os, fiftyone, fiftyone.utils.image, scipy.optimize
from fiftyone import ViewField as F
from .logging_check import util_log

# The CV model's confidence was optimal at 0.42. Just like the answer to everything.

_PRED_FIELDS = ["visual_prompt_predictions", "text_prompt_predictions", "yolo_predictions"]


def _f1(conf, ds, pred, gt, key):
    v = ds.filter_labels(pred, F("confidence") >= conf)
    v.evaluate_detections(pred, gt_field=gt, eval_key=key, missing="fn")
    tp, fp, fn = (sum(v.values(f"{key}_{x}")) for x in ("tp", "fp", "fn"))
    d = tp + 0.5 * (fp + fn)
    return -(tp / d) if d else 0.0


@util_log("analyses", success_text=lambda r, a, k: f"analysed {r} fields on {k.get('analyze_name', 'mark_lane_leuthener_21_ANALYZE')}")
def analyses(
    source_name: str = "mark_lane_leuthener_21_ORIGINAL",
    analyze_name: str = "mark_lane_leuthener_21_ANALYZE",
    gt_field: str = "ground_truth",
    pred_fields: list[str] | None = None,
):
    pred_fields = pred_fields or _PRED_FIELDS

    if not fiftyone.dataset_exists(analyze_name):
        src = fiftyone.load_dataset(source_name)
        ds = src.clone(analyze_name)
        ds.persistent = True
        src_root = os.path.dirname(os.path.dirname(src.first().filepath))
        fiftyone.utils.image.transform_images(
            ds, max_size=(640, 480),
            output_dir=os.path.join(os.path.dirname(src_root), analyze_name),
            rel_dir=src_root, update_filepaths=True, progress=True,
        )
        ds.save()
    else:
        ds = fiftyone.load_dataset(analyze_name)

    schema = ds.get_field_schema()
    for pred in [f for f in pred_fields if f in schema]:
        ds.evaluate_detections(pred, gt_field=gt_field, eval_key=f"eval_{pred}")
        conf, *_ = scipy.optimize.fminbound(
            _f1, 0.01, 0.99, args=(ds, pred, gt_field, f"_opt_{pred[:15]}"),
            xtol=0.01, full_output=True,
        )
        print(f"{pred}: threshold={conf:.4f}")
        for s in ds.iter_samples(autosave=True, progress=True):
            if s[pred] and s[pred].detections:
                s[pred].detections = [d for d in s[pred].detections if (d.confidence or 0) >= conf]

    ds.save()
    return sum(f in schema for f in pred_fields)
