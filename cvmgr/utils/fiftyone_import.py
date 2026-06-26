import fiftyone
import pathlib
import numpy as np
import cv2
from .sam3_visual_segmentation import sam3_visual_segmentation
from .sam2_visual_segmentation import sam2_visual_segmentation
from .fiftyone_replace import fiftyone_replace
from .logging_check import util_log


def _polylines_to_detections(dataset, src_field: str, dst_field: str):
    dataset.compute_metadata()
    for sample in dataset.iter_samples(autosave=True):
        pl = sample.get_field(src_field)
        if not pl or not pl.polylines:
            sample[dst_field] = fiftyone.core.labels.Detections(detections=[])
            continue
        w, h = sample.metadata.width, sample.metadata.height
        dets = []
        for p in pl.polylines:
            contours = [np.array([[round(x * w), round(y * h)] for x, y in c], np.int32) for c in p.points if len(c) >= 3]
            if not contours:
                continue
            stacked = np.vstack(contours)
            x0, y0 = stacked[:, 0].min(), stacked[:, 1].min()
            x1, y1 = stacked[:, 0].max(), stacked[:, 1].max()
            if x1 == x0 or y1 == y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [c - np.array([x0, y0]) for c in contours], 1)
            dets.append(fiftyone.core.labels.Detection(
                label=p.label,
                bounding_box=[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                mask=mask.astype(bool),
            ))
        sample[dst_field] = fiftyone.core.labels.Detections(detections=dets)


@util_log("fiftyone_import", success_text=lambda result, args, kwargs: "dataset_exists OR replaced")
def fiftyone_import(dataset_name: str, config: dict, replace: bool = False, gpu: str = "0"):

    if fiftyone.dataset_exists(dataset_name) and not replace:
        return True
    if fiftyone.dataset_exists(dataset_name) and replace:
        dataset = fiftyone.load_dataset(dataset_name)
        dataset.delete()

    import_path = pathlib.Path(fiftyone.config.default_dataset_dir) / dataset_name
    
    dataset = fiftyone.Dataset(dataset_name, persistent=True)
    import_label_field = "ground_truth_polylines" if config.get("type") == "polylines" else "ground_truth"
    for split in config.get("download_splits", []):
        split_path = import_path / "images" / split
        if split_path.exists() and any(split_path.iterdir()):
            dataset.add_dir(
                dataset_dir=str(import_path),
                dataset_type=fiftyone.types.YOLOv5Dataset,
                label_type=config.get("type"),
                label_field=import_label_field,
                split=split,
                tags=split,
                seed=42,
            )
            
    if config.get("samples_per_split"):
        view = dataset.take(config.get("samples_per_split")*len(config.get("download_splits", [])))
        dataset.delete_samples(dataset.exclude(view))
    if config.get("type") == "detections":
        dataset = sam3_visual_segmentation(dataset, recalculate=False, gpu=gpu)
        if dataset is False:
            return False
    elif config.get("type") == "polylines":
        _polylines_to_detections(dataset, "ground_truth_polylines", "ground_truth")
    
    dataset.save()
    return fiftyone.dataset_exists(dataset_name)