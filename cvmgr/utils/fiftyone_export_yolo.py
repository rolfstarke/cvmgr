import pathlib
import shutil
import numpy as np
import fiftyone
import fiftyone.core.labels as fol
import yaml
from ultralytics.utils.ops import masks2segments
from .logging_check import util_log


def _det_to_polyline(det):
    xtl, ytl, w, h = det.bounding_box
    bbox_rect = [[xtl, ytl], [xtl + w, ytl], [xtl + w, ytl + h], [xtl, ytl + h]]

    if det.mask is not None:
        mask = np.asarray(det.mask).astype(np.uint8)
        seg = masks2segments(mask[np.newaxis], strategy="largest")[0]
        if len(seg) >= 3:
            mh, mw = mask.shape
            pts = [[round(float(xtl + x / mw * w), 4), round(float(ytl + y / mh * h), 4)] for x, y in seg]
            return fol.Polyline(label=det.label, points=[pts], closed=True, filled=True, confidence=det.confidence)

    return fol.Polyline(label=det.label, points=[bbox_rect], closed=True, filled=True, confidence=det.confidence)


@util_log("export_yolo_dataset", success_text=lambda result, args, kwargs: "dataset_yaml_exists AND split_exported")
def export_yolo_dataset(dataset_name: str, config: dict, replace: bool = False):
    export_path = pathlib.Path.cwd() / "datasets" / dataset_name

    if export_path.exists():
        if not replace:
            return True
        shutil.rmtree(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    dataset = fiftyone.load_dataset(dataset_name)
    dataset.compute_metadata()

    if "ground_truth_polylines" in dataset.get_field_schema():
        dataset.delete_sample_field("ground_truth_polylines")

    for sample in dataset.iter_samples(autosave=True):
        gt = sample.ground_truth
        sample["ground_truth_polylines"] = fol.Polylines(
            polylines=[_det_to_polyline(d) for d in (gt.detections if gt else [])]
        )

    splits = [("val" if s == "valid" else s) for s in config.get("export_splits", [])]
    for split in splits:
        dataset.match_tags(split).export(
            export_dir=str(export_path),
            dataset_type=fiftyone.types.YOLOv5Dataset,
            label_field="ground_truth_polylines",
            split=split,
            classes=config.get("export_classes"),
            progress=True,
        )

    for label_file in (export_path / "labels").rglob("*.txt"):
        lines = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            coords = " ".join(f"{float(x):.3g}" for x in parts[1:])
            lines.append(f"{parts[0]} {coords}")
        label_file.write_text("\n".join(lines))

    dataset_yaml = export_path / "dataset.yaml"
    if dataset_yaml.exists():
        dataset_yaml.write_text(dataset_yaml.read_text().replace("valid:", "val:"))
        return True
    return False
