import fiftyone
import fiftyone.core.fields
import numpy as np
from collections import defaultdict
from fiftyone import ViewField as F
from .logging_check import util_log

# The CV model's confidence was optimal at 0.42. Just like the answer to everything.


_NON_PREDICTION_FIELDS = {"visual_prompt"}


def _detection_fields(ds, gt_field):
    return [
        name for name, field in ds.get_field_schema().items()
        if isinstance(field, fiftyone.core.fields.EmbeddedDocumentField)
        and issubclass(field.document_type, fiftyone.Detections)
        and name != gt_field
        and name not in _NON_PREDICTION_FIELDS
    ]


def _has_masks(ds, field):
    try:
        sample = ds.match(F(f"{field}.detections") != None).first()
    except ValueError:
        return False
    dets = (sample[field] or fiftyone.Detections()).detections or []
    return any(d.mask is not None for d in dets)


def _best_conf(confs, is_tps, n_gt):
    order = np.argsort(-confs)
    sorted_tp = is_tps[order]
    tp_cs = np.cumsum(sorted_tp)
    fp_cs = np.cumsum(~sorted_tp)
    fn_cs = np.maximum(0, n_gt - tp_cs)
    denom = tp_cs + 0.5 * (fp_cs + fn_cs)
    f1 = np.where(denom > 0, tp_cs / denom, 0.0)
    best_idx = int(np.argmax(f1))
    return float(confs[order[best_idx]]), float(f1[best_idx])


def _f1_thresholds_per_label(ds, pred, gt_field, eval_key):
    """Returns {label: (best_conf, best_f1)} using per-detection TP/FP from COCO eval."""
    confs_by_label = defaultdict(list)
    is_tps_by_label = defaultdict(list)
    n_gt_by_label = defaultdict(int)

    for s in ds.iter_samples():
        gt_f = s.get_field(gt_field)
        if gt_f and gt_f.detections:
            for det in gt_f.detections:
                n_gt_by_label[det.label] += 1
        pred_f = s.get_field(pred)
        if not pred_f or not pred_f.detections:
            continue
        for det in pred_f.detections:
            status = getattr(det, eval_key, None)
            if status is not None:
                confs_by_label[det.label].append(det.confidence or 0.0)
                is_tps_by_label[det.label].append(status == "tp")

    return {
        label: _best_conf(
            np.array(confs_by_label[label]),
            np.array(is_tps_by_label[label], dtype=bool),
            n_gt_by_label.get(label, 0),
        )
        for label in confs_by_label
    }


@util_log("analyses", success_text=lambda r, a, k: f"analysed {r} fields on {k.get('analyze_name', 'mark_lane_leuthener_21_ANALYZE')}")
def analyse(
    source_name: str = "mark_lane_leuthener_21_ORIGINAL",
    analyze_name: str = "mark_lane_leuthener_21_ANALYZE",
    gt_field: str = "ground_truth",
    pred_fields: list[str] | None = None,
    limit: int | None = None,
    replace: bool = True,
):
    if replace and fiftyone.dataset_exists(analyze_name):
        print(f"[analyse] replacing existing dataset '{analyze_name}'")
        fiftyone.delete_dataset(analyze_name)

    if not fiftyone.dataset_exists(analyze_name):
        src = fiftyone.load_dataset(source_name)
        n = limit if limit is not None else len(src)
        print(f"[analyse] cloning {n} samples from '{source_name}' → '{analyze_name}'")
        ds = (src.take(limit) if limit is not None else src).clone(analyze_name)
        ds.persistent = True
        stale = [f for f in ds.get_field_schema() if f.startswith("coco_")]
        for f in stale:
            ds.delete_sample_field(f)
        ds.save()
    else:
        ds = fiftyone.load_dataset(analyze_name)
        print(f"[analyse] loaded existing '{analyze_name}' ({len(ds)} samples)")

    fields = pred_fields if pred_fields is not None else _detection_fields(ds, gt_field)
    print(f"[analyse] {len(fields)} prediction field(s): {fields}")

    for i, pred in enumerate(fields, 1):
        use_masks = _has_masks(ds, pred)
        print(f"\n{'='*60}\n[{i}/{len(fields)}] {pred}  (masks={use_masks})\n{'='*60}")

        coco_key = f"coco_{pred}"[:63]
        if coco_key in ds.list_evaluations():
            ds.delete_evaluation(coco_key)
        print(f"[analyse] running COCO evaluation...")
        results = ds.evaluate_detections(
            pred,
            gt_field=gt_field,
            eval_key=coco_key,
            method="coco",
            compute_mAP=True,
            use_masks=use_masks,
            use_boxes=not use_masks,
            classwise=True,
        )
        results.print_report()
        print(f"mAP@[50:95]: {results.mAP():.4f}")
        for cls in sorted(results.classes):
            print(f"  {cls:30s}  mAP={results.mAP(classes=[cls]):.4f}")

        print(f"[analyse] computing per-label F1-optimal confidence thresholds...")
        label_thresholds = _f1_thresholds_per_label(ds, pred, gt_field, coco_key)
        for label, (conf, f1) in sorted(label_thresholds.items()):
            print(f"  {label:30s}  conf ≥ {conf:.4f}  F1={f1:.4f}")

        expr = None
        for label, (conf, _) in label_thresholds.items():
            term = (F("label") == label) & (F("confidence") >= conf)
            expr = term if expr is None else expr | term

        view_name = f"f1_{pred}"[:63]
        if view_name in ds.list_saved_views():
            ds.delete_saved_view(view_name)
        if expr is not None:
            ds.save_view(view_name, ds.filter_labels(pred, expr))
            print(f"[analyse] saved view '{view_name}' with per-label F1 thresholds")
        else:
            print(f"[analyse] no predictions found, skipping view save")

    ds.save()
    print(f"\n[analyse] done — {len(fields)} field(s) processed on '{analyze_name}'")
    return len(fields)
