import fiftyone as fiftyone
import random
from .logging_check import util_log

_SPLIT_TAGS = {"train", "test", "val", "valid", "validation"}

@util_log("redistribute_splits", success_text=lambda result, args, kwargs: "split_counts == dataset_len")
def redistribute_splits(dataset_name: str):

    dataset = fiftyone.load_dataset(dataset_name)
    dataset.untag_samples(list(_SPLIT_TAGS))

    rng = random.Random(42)
    split_counts = {"train": 0, "val": 0}

    for sample in dataset.iter_samples(progress=True):
        draw = rng.random()
        if draw < 0.85:
            split = "train"
        else:
            split = "val"

        sample.tags = [t for t in sample.tags if t not in _SPLIT_TAGS] + [split]
        sample.save()
        split_counts[split] += 1

    dataset.persistent = True
    dataset.save()
    return sum(split_counts.values()) == len(dataset)
