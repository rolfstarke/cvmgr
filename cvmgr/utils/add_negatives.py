import pathlib
import yaml
import fiftyone
from fiftyone import ViewField as F

from .logging_check import util_log

_datasets_path = pathlib.Path(__file__).parent.parent / "configs" / "datasets.yaml"


@util_log("add_negatives", success_text=lambda result, args, kwargs: "negatives_added")
def add_negatives(dataset_name: str):
    dataset = fiftyone.load_dataset(dataset_name)
    target_classes = dataset.distinct("ground_truth.detections.label")

    max_negatives = int(len(dataset) * 0.2)
    if max_negatives <= 0:
        return True

    with _datasets_path.open() as f:
        datasets_cfg = yaml.safe_load(f)
    hard_negatives = datasets_cfg.get("oi_v7_custom_negative", {}).get("hard_negatives", {})

    # find hard-negative classes configured for the target classes in this dataset
    confusable_classes = [
        negative_cls for export_cls, negative_cls in hard_negatives.items() if export_cls in target_classes
    ]

    negatives = fiftyone.load_dataset("oi_v7_custom_negative")
    if confusable_classes:
        view = negatives.match(
            (
                F("ground_truth.detections.label").contains(confusable_classes, all=False)
                | F("sam3_predictions.detections.label").contains(confusable_classes, all=False)
            )
            & ~F("ground_truth.detections.label").contains(target_classes, all=False)
            & ~F("sam3_predictions.detections.label").contains(target_classes, all=False)
        ).take(max_negatives, seed=42)
    else:
        view = negatives.match(
            ~F("ground_truth.detections.label").contains(target_classes, all=False)
            & ~F("sam3_predictions.detections.label").contains(target_classes, all=False)
        ).take(max_negatives, seed=42)

    existing_filepaths = set(dataset.values("filepath"))
    novel = view.match(~F("filepath").is_in(list(existing_filepaths)))
    dataset.add_samples(novel.exclude_fields("ground_truth"))
    dataset.save()
    return True
