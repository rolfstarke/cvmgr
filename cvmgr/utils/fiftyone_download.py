
import fiftyone
from .sam3_visual_segmentation import sam3_visual_segmentation
import fiftyone.utils.iou
from .logging_check import util_log


@util_log("fiftyone_download", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "dataset_exists OR filtered")
def fiftyone_download(dataset_name: str, config: dict, gpu: str = "0"):

    samples_per_class = config.get("samples_per_class")
    download_classes = config.get("download_classes")

    if samples_per_class and download_classes:
        tmp_name = f"__tmp_{dataset_name}_merge"
        if fiftyone.dataset_exists(tmp_name):
            fiftyone.delete_dataset(tmp_name)
        merged = fiftyone.Dataset(tmp_name)
        for i, cls in enumerate(download_classes):
            cls_tmp_name = f"__tmp_{dataset_name}_cls_{i}"
            if fiftyone.dataset_exists(cls_tmp_name):
                fiftyone.delete_dataset(cls_tmp_name)
            cls_dataset = fiftyone.zoo.load_zoo_dataset(
                name_or_url=config.get("origin"),
                splits=config.get("download_splits"),
                label_types=config.get("type"),
                classes=[cls],
                max_samples=samples_per_class,
                dataset_name=cls_tmp_name,
            )
            merged.merge_samples(cls_dataset)
            fiftyone.delete_dataset(cls_tmp_name)
        dataset = merged
    else:
        dataset = fiftyone.zoo.load_zoo_dataset(
            name_or_url=config.get("origin"),
            splits=config.get("download_splits"),
            label_types=config.get("type"),
            classes=download_classes,
            max_samples=config.get("samples_per_split"),
        )

    if download_classes:
        dataset = dataset.filter_labels(
            "ground_truth",
            fiftyone.ViewField("label").is_in(download_classes)
        )

    dataset = dataset.clone(dataset_name)
    dataset.persistent = True

    if config.get("label_map"):
        dataset.map_labels("ground_truth", config["label_map"]).save()

    if samples_per_class and fiftyone.dataset_exists(tmp_name):
        fiftyone.delete_dataset(tmp_name)

    if config.get("type") == "detections":
        dataset = sam3_visual_segmentation(dataset=dataset, recalculate=False, gpu=gpu)
        if dataset is False:
            return False

    if not fiftyone.dataset_exists(dataset_name):
        return False

    return dataset

