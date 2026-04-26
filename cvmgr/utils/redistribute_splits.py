import fiftyone as fiftyone
import random
import logging

def redistribute_splits(dataset_name: str):

    dataset = fiftyone.load_dataset(dataset_name)

    # Remove only split-related tags to avoid clobbering unrelated metadata tags.
    dataset.untag_samples(["train", "test", "val", "validation"])

    rng = random.Random(42)
    split_counts = {"train": 0, "test": 0, "val": 0}

    # Assign tags one sample at a time so no large ID list is sent to MongoDB.
    for sample in dataset.iter_samples(progress=True):
        draw = rng.random()
        if draw < 0.7:
            split = "train"
        elif draw < 0.9:
            split = "test"
        else:
            split = "val"

        tags = [t for t in sample.tags if t not in {"train", "test", "val", "validation"}]
        tags.append(split)
        sample.tags = tags
        sample.save()
        split_counts[split] += 1
    
    dataset.persistent = True
    dataset.save()
    logging.info(f"Splits after redistribution: {split_counts}")
