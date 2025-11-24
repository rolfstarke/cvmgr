

def fiftyone_download(name: str, max_samples: int, origin: str, type: str, splits: list, classes: list):

    dataset = fiftyone.zoo.load_zoo_dataset(
        name_or_url=origin,
        splits=splits,
        label_types=type,
        classes=classes,
        max_samples=max_samples,
    )

    dataset.name = f"{name}"+f"_{max_samples}"
    dataset.save()
    

