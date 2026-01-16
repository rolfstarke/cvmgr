import fiftyone
from .fiftyone_download import fiftyone_download
from .fiftyone_replace import fiftyone_replace
from .roboflow_download import roboflow_download
import logging
logger = logging.getLogger('cvmgr')

def fetch_dataset(dataset_name: str, config: dict, replace: bool = False):

    if config.get("multipart"):
        if not fiftyone_replace(dataset_name, replace):
            return
        dataset = fiftyone.Dataset(dataset_name)
        dataset.persistent = True
        for key, value in config.items():
            if key != "multipart":  # Skip the multipart flag
                if not fiftyone_replace(key, replace):
                    return
                if value.get("host") == "fiftyone_zoo":
                    dataset.merge_samples(fiftyone_download(key, value), progress=True)
                    dataset.save()
                
    else:
        if not fiftyone_replace(dataset_name, replace):
            return
        if config.get("host") == "fiftyone_zoo":
            dataset = fiftyone_download(dataset_name, config)
            dataset.save()
        if config.get("host") == "roboflow":
            roboflow_download(dataset_name, config)

