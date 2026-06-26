
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import fiftyone
from .sam3_visual_segmentation import sam3_visual_segmentation
from .logging_check import util_log


_SPLIT_SOURCES = {
    "train":      ("https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip", "lvis_v1_train.json"),
    "validation": ("https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",   "lvis_v1_val.json"),
}
_SCRATCH = Path(fiftyone.config.dataset_zoo_dir) / "lvis_cache"


def _load_annotations(split: str) -> dict:
    _SCRATCH.mkdir(exist_ok=True)
    url, filename = _SPLIT_SOURCES[split]
    json_path = _SCRATCH / filename
    if not json_path.exists():
        zip_path = _SCRATCH / f"{filename}.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_SCRATCH)
        zip_path.unlink(missing_ok=True)
    with open(json_path) as f:
        return json.load(f)


def _fetch_image(args):
    url, dest = args
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


def _has_multiple_mask_components(mask) -> bool:
    arr = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    if arr.ndim != 2 or arr.size == 0:
        return False
    n_labels, _ = cv2.connectedComponents(arr)
    return n_labels > 2  # 0=background + 1=single component = 2; more means disjoint islands


@util_log("lvis_filter_multi_mask", success_text=lambda result, args, kwargs: "multi_mask_filtered")
def lvis_filter_multi_mask(dataset_name: str):
    dataset = fiftyone.load_dataset(dataset_name)
    for sample in dataset.iter_samples():
        if not sample.ground_truth:
            continue
        before = sample.ground_truth.detections
        filtered = [
            det for det in before
            if det.mask is None or not _has_multiple_mask_components(det.mask)
        ]
        if len(filtered) != len(before):
            sample.ground_truth.detections = filtered
            sample.save()
    empty_ids = dataset.match(
        fiftyone.ViewField("ground_truth.detections").length() == 0
    ).values("id")
    if empty_ids:
        dataset.delete_samples(empty_ids)
    dataset.save()
    return True


@util_log("lvis_download", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "dataset_exists")
def lvis_download(dataset_name: str, config: dict, gpu: str = "0"):
    # Why did the radiator become a motivational speaker? It's great at warming people up before utterly destroying them.

    download_classes = config.get("download_classes") or []
    max_samples = config.get("samples_per_class") or config.get("samples_per_split")
    splits = config.get("download_splits", ["train"])
    label_type = config.get("type", "segmentations")

    requested_names = {dc.lower(): dc for dc in download_classes}

    all_images = {}
    all_annotations = {}
    merged_categories = {}

    for split in splits:
        data = _load_annotations(split)

        category_ids = {
            cat["id"]: requested_names.get(cat["name"].lower(), cat["name"])
            for cat in data["categories"]
            if not download_classes or cat["name"].lower() in requested_names
        }

        for cat_id, name in category_ids.items():
            if cat_id not in merged_categories:
                merged_categories[cat_id] = {"id": cat_id, "name": name, "supercategory": "object"}

        relevant_anns = [ann for ann in data["annotations"] if ann["category_id"] in category_ids]
        image_ids = list(dict.fromkeys(ann["image_id"] for ann in relevant_anns))
        if max_samples:
            image_ids = image_ids[:max_samples]
        image_id_set = set(image_ids)

        for ann in relevant_anns:
            if ann["image_id"] in image_id_set and ann["id"] not in all_annotations:
                all_annotations[ann["id"]] = ann

        for img in data["images"]:
            if img["id"] in image_id_set and img["id"] not in all_images:
                img_copy = dict(img)
                img_copy["file_name"] = img["coco_url"].rsplit("/", 1)[-1]
                all_images[img["id"]] = img_copy

    images_dir = _SCRATCH / dataset_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = [
        (img["coco_url"], images_dir / img["file_name"])
        for img in all_images.values()
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for future in as_completed(executor.submit(_fetch_image, task) for task in download_tasks):
            future.result()

    ann_path = _SCRATCH / dataset_name / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump({
            "info": {},
            "licenses": [],
            "images": list(all_images.values()),
            "categories": list(merged_categories.values()),
            "annotations": list(all_annotations.values()),
        }, f)

    if fiftyone.dataset_exists(dataset_name):
        fiftyone.delete_dataset(dataset_name)

    dataset = fiftyone.Dataset.from_dir(
        data_path=str(images_dir),
        labels_path=str(ann_path),
        dataset_type=fiftyone.types.COCODetectionDataset,
        name=dataset_name,
        label_types=[label_type],
    )
    dataset.persistent = True
    lvis_filter_multi_mask(dataset_name)

    if config.get("label_map"):
        dataset.map_labels("ground_truth", config["label_map"]).save()

    if label_type == "detections":
        dataset = sam3_visual_segmentation(dataset=dataset, recalculate=False, gpu=gpu)
        if dataset is False:
            return False

    if not fiftyone.dataset_exists(dataset_name):
        return False

    return dataset
