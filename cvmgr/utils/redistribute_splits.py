import fiftyone as fiftyone
import fiftyone.utils.random as four

def redistribute_splits(dataset_name: str):

    dataset = fiftyone.load_dataset(dataset_name)

    dataset.untag_samples(dataset.distinct("tags"))
    validation_samples = dataset.match_tags("validation")
    validation_samples.tag_samples("val")
    validation_samples.untag_samples("validation")

    four.random_split(
        dataset,
        {"train": 0.7, "test": 0.2, "val": 0.1}
        )
    
    dataset.persistent = True
    dataset.save()
    print(f"Splits after redistribution: {dataset.count_sample_tags()}")
