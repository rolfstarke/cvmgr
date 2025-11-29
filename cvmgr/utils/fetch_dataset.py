import fiftyone
from .fiftyone_download import fiftyone_download

def fetch_dataset(dataset_name: str, config: dict, replace: bool = False):

    if fiftyone.dataset_exists(dataset_name):
        if replace:
            fiftyone.delete_dataset(dataset_name)
        else:
            print(f"Dataset {dataset_name} already exists. Skipping download.")
            return

    if config.get("host") == "fiftyone_zoo":
        fiftyone_download(dataset_name, config)
    else:
        raise ValueError(f"Unsupported host: {config.get('host')}")