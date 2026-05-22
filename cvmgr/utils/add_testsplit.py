import fiftyone
from fiftyone import ViewField as F

from .logging_check import util_log


TEST_DATASET = "mark_lane_leuthener_custom_21_hd"


@util_log("add_testsplit", success_text=lambda result, args, kwargs: f"added_samples={result}")
def add_testsplit(dataset_name: str) -> int:
    dataset = fiftyone.load_dataset(dataset_name)
    test_dataset = fiftyone.load_dataset(TEST_DATASET)

    classes = dataset.distinct("ground_truth.detections.label")
    view = test_dataset.filter_labels("ground_truth", F("label").is_in(classes))

    for sample in view:
        dataset.add_sample(sample)

    dataset.save()
    return len(view)
