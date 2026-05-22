import fiftyone
from .fiftyone_download import fiftyone_download
from .fiftyone_replace import fiftyone_replace
from .roboflow_download import roboflow_download
from .logging_check import util_log

@util_log("fetch_dataset", success_text=lambda result, args, kwargs: "replace_ok AND source_ok")
def fetch_dataset(dataset_name: str, config: dict, replace: bool = False):

    if config.get("multipart"):
        if not fiftyone_replace(dataset_name, replace):
            return False
        dataset = fiftyone.Dataset(dataset_name)
        dataset.persistent = True
        for key, value in config.items():
            if key != "multipart":  # Skip the multipart flag
                if not fiftyone_replace(key, replace):
                    return False
                if value.get("host") == "fiftyone_zoo":
                    sub_dataset = fiftyone_download(key, value)
                    if sub_dataset is False:
                        return False
                    dataset.merge_samples(sub_dataset, progress=True)
                    dataset.save()
                
    else:
        if not fiftyone_replace(dataset_name, replace):
            return False
        if config.get("host") == "fiftyone_zoo":
            dataset = fiftyone_download(dataset_name, config)
            if dataset is False:
                return False
            dataset.save()
        if config.get("host") == "roboflow":
            if not roboflow_download(dataset_name, config):
                return False

    return True

