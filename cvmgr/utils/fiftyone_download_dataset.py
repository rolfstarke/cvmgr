
import fiftyone

def fiftyone_download(name: str, config: dict, replace: bool = False):

    if config.get("samples_per_split") is None:
        name_with_samplenr = name
    else:
        print(f"Downloading {config.get('samples_per_split')*len(config.get('splits'))} samples.")
        name_with_samplenr = f"{name}"+f"_{config.get('samples_per_split')*len(config.get('splits'))}"

    if fiftyone.dataset_exists(name_with_samplenr):
        if replace:
            fiftyone.delete_dataset(name_with_samplenr)
        else:
            print(f"Dataset {name_with_samplenr} already exists. Skipping download.")
            return


    dataset = fiftyone.zoo.load_zoo_dataset(
        name_or_url=config.get("origin"),
        splits=config.get("splits"),
        label_types=config.get("type"),
        classes=config.get("classes"),
        max_samples=config.get("samples_per_split"),
    )

    dataset.name = name_with_samplenr
    dataset.save()

    if fiftyone.dataset_exists(name_with_samplenr):
        print(f"Dataset {name_with_samplenr} with {len(dataset)} samples downloaded successfully.")
    

