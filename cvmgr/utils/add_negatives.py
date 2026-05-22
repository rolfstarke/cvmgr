import fiftyone
from fiftyone import ViewField as F

from .logging_check import util_log

@util_log("add_negatives", success_text=lambda result, args, kwargs: "negatives_added")
def add_negatives(dataset_name: str):
    dataset = fiftyone.load_dataset(dataset_name)
    existing_classes = dataset.distinct("ground_truth.detections.label")

    max_negatives = int(len(dataset) * 0.2)
    if max_negatives <= 0:
        return True

    negatives = fiftyone.load_dataset("oi_v7_custom_negative")
    view = negatives.match(
        ~F("ground_truth.detections.label").contains(existing_classes, all=False)
        & ~F("sam3_predictions.detections.label").contains(existing_classes, all=False)
    ).take(max_negatives, seed=42)

    for s in view:
        s.clear_field("ground_truth")
        dataset.add_sample(s)
    dataset.save()
    return True
