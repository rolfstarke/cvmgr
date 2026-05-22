import fiftyone
from fiftyone import ViewField as F

from .logging_check import util_log

@util_log("add_negatives", success_text=lambda result, args, kwargs: "negatives_added")
def add_negatives(dataset_name: str):
    dataset = fiftyone.load_dataset(dataset_name)
    existing_classes = dataset.distinct("ground_truth.detections.label")

    negatives = fiftyone.load_dataset("oi_v7_custom_negative")
    view = negatives.match(
        ~F("ground_truth.detections.label").contains(existing_classes, all=False)
        & ~F("sam3_predictions.detections.label").contains(existing_classes, all=False)
    )

    for s in view:
        s.clear_field("ground_truth")
        dataset.add_sample(s)
    dataset.save()
    return True
