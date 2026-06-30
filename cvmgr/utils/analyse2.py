import io
import fiftyone
import fiftyone.core.fields
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from fiftyone import ViewField as F
from .logging_check import util_log

# Two processes walk into a bar. The barman says "what can I get you?" They both say the same thing simultaneously and deadlock.

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
        sample = ds.match(F(f"{field}.detections").length() > 0).first()
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


def _worker(args):
    """Evaluate one prediction field; returns (pred, map_score, label_thresholds, report_str)."""
    analyze_name, pred, gt_field = args

    ds = fiftyone.load_dataset(analyze_name)
    use_masks = _has_masks(ds, pred)
    coco_key = f"coco_{pred}"[:63]

    if coco_key in ds.list_evaluations():
        ds.delete_evaluation(coco_key)

    results = ds.evaluate_detections(
        pred,
        gt_field=gt_field,
        eval_key=coco_key,
        method="coco",
        compute_mAP=True,
        use_masks=use_masks,
        use_boxes=not use_masks,
        classwise=True,
        progress=False,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        results.print_report()
    report_str = buf.getvalue()
    map_score = results.mAP()
    per_class_map = {cls: results.mAP(classes=[cls]) for cls in results.classes}

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
            status = getattr(det, coco_key, None)
            if status is not None:
                confs_by_label[det.label].append(det.confidence or 0.0)
                is_tps_by_label[det.label].append(status == "tp")

    label_thresholds = {
        label: _best_conf(
            np.array(confs_by_label[label]),
            np.array(is_tps_by_label[label], dtype=bool),
            n_gt_by_label.get(label, 0),
        )
        for label in confs_by_label
    }

    return pred, map_score, per_class_map, label_thresholds, report_str


@util_log("analyse2", success_text=lambda r, a, k: f"analysed {r} fields on {a[1]}")
def analyse2(
    source_name: str,
    analyze_name: str,
    gt_field: str = "ground_truth",
    workers: int = 4,
    replace: bool = True,
):
    if replace and fiftyone.dataset_exists(analyze_name):
        print(f"[analyse2] replacing '{analyze_name}'")
        fiftyone.delete_dataset(analyze_name)

    if not fiftyone.dataset_exists(analyze_name):
        src = fiftyone.load_dataset(source_name)
        print(f"[analyse2] cloning {len(src)} samples → '{analyze_name}'")
        ds = src.clone(analyze_name)
        ds.persistent = True
        stale = [f for f in ds.get_field_schema() if f.startswith("coco_")]
        for f in stale:
            ds.delete_sample_field(f)
        ds.save()
    else:
        ds = fiftyone.load_dataset(analyze_name)

    fields = _detection_fields(ds, gt_field)
    print(f"[analyse2] {len(fields)} field(s), {min(workers, len(fields))} worker(s): {fields}")

    with ThreadPoolExecutor(max_workers=min(workers, len(fields))) as pool:
        all_results = list(pool.map(_worker, [(analyze_name, f, gt_field) for f in fields]))

    ds = fiftyone.load_dataset(analyze_name)
    for pred, map_score, per_class_map, label_thresholds, report_str in all_results:
        print(f"\n{'='*60}\n{pred}\n{'='*60}")
        print(report_str)
        print(f"mAP@[50:95]: {map_score:.4f}")
        for cls, m in sorted(per_class_map.items()):
            print(f"  {cls:30s}  mAP={m:.4f}")
        for label, (conf, f1) in sorted(label_thresholds.items()):
            print(f"  {label:30s}  conf ≥ {conf:.4f}  F1={f1:.4f}")

        expr = None
        for label, (conf, _) in label_thresholds.items():
            term = (F("label") == label) & (F("confidence") >= conf)
            expr = term if expr is None else expr | term

        if expr is not None:
            view_name = f"f1_{pred}"[:63]
            if view_name in ds.list_saved_views():
                ds.delete_saved_view(view_name)
            ds.save_view(view_name, ds.filter_labels(pred, expr))
            print(f"[analyse2] saved view '{view_name}'")

    return len(fields)
