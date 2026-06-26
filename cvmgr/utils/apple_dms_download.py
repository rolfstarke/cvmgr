import gzip
import json
import pathlib
import urllib.request
import zipfile

import tqdm

import numpy as np
import scipy.ndimage
from PIL import Image
import fiftyone
import fiftyone.core.labels

from .logging_check import util_log


_LABELS_ZIP_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_labels.zip"
_CACHE = pathlib.Path.home() / ".cache" / "apple_dms"


@util_log("apple_dms_download", success_text=lambda result, args, kwargs: f"dataset_created={result}")
def apple_dms_download(dataset_name: str, config: dict) -> bool:
    # Why having sex with your carpet is illegal but selling it isn't is a
    # question no legal scholar has satisfactorily answered.
    dms_class_id: int = config["dms_class_id"]
    download_classes = config["download_classes"]

    _CACHE.mkdir(parents=True, exist_ok=True)
    labels_dir = _ensure_labels_archive()
    info = _load_info(labels_dir)

    if fiftyone.dataset_exists(dataset_name):
        fiftyone.delete_dataset(dataset_name)
    dataset = fiftyone.Dataset(dataset_name)
    dataset.persistent = True

    for datum in tqdm.tqdm(info, desc=f"{dataset_name} download", unit="img"):
        label_path = labels_dir / datum["label_path"]
        if not label_path.exists():
            continue

        label_map = np.array(Image.open(label_path))
        class_mask = label_map == dms_class_id
        if not class_mask.any():
            continue

        image_path = _ensure_image(datum)
        if image_path is None:
            continue

        split = _split_from_path(datum["image_path"])
        detections = _mask_to_detections(class_mask, download_classes)
        sample = fiftyone.Sample(filepath=str(image_path))
        sample.tags = [split]
        sample["ground_truth"] = fiftyone.core.labels.Detections(detections=detections)
        dataset.add_sample(sample)

    dataset.save()
    return fiftyone.dataset_exists(dataset_name)


def _ensure_labels_archive() -> pathlib.Path:
    zip_path = _CACHE / "dms_v1_labels.zip"
    with tqdm.tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="DMS labels") as bar:
        def _reporthook(count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)
        urllib.request.urlretrieve(_LABELS_ZIP_URL, zip_path, reporthook=_reporthook)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        with tqdm.tqdm(members, desc="Extracting", unit="file") as bar:
            for member in bar:
                zf.extract(member, _CACHE / "labels")
    return _CACHE / "labels" / "DMS_v1"


def _load_info(labels_dir: pathlib.Path) -> list:
    info_path = labels_dir / "info.json.gz"
    with gzip.open(info_path, "rb") as f:
        return json.loads(f.read())


def _ensure_image(datum: dict) -> pathlib.Path | None:
    rel = datum["image_path"]
    image_path = _CACHE / rel
    if image_path.exists():
        return image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    url = datum["openimages_metadata"]["OriginalURL"]
    try:
        urllib.request.urlretrieve(url, image_path)
        return image_path
    except Exception:
        return None


def _split_from_path(image_path: str) -> str:
    parts = pathlib.PurePosixPath(image_path).parts
    raw = parts[1] if len(parts) > 1 else "train"
    return "val" if raw == "validation" else raw


def _mask_to_detections(
    class_mask: np.ndarray, label: str
) -> list[fiftyone.core.labels.Detection]:
    labeled, n = scipy.ndimage.label(class_mask)
    h, w = class_mask.shape
    detections = []
    for i in range(1, n + 1):
        inst = labeled == i
        rows = np.any(inst, axis=1)
        cols = np.any(inst, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = [cmin / w, rmin / h, (cmax - cmin + 1) / w, (rmax - rmin + 1) / h]
        detections.append(fiftyone.core.labels.Detection(
            label=label,
            bounding_box=bbox,
            mask=inst[rmin:rmax + 1, cmin:cmax + 1],
        ))
    return detections
