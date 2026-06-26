import fiftyone
from .fiftyone_download import fiftyone_download
from .fiftyone_replace import fiftyone_replace
from .roboflow_download import roboflow_download
from .apple_dms_download import apple_dms_download
from .lvis_download import lvis_download
from .logging_check import util_log

@util_log("fetch_dataset", success_text=lambda result, args, kwargs: "replace_ok AND source_ok")
def fetch_dataset(dataset_name: str, config: dict, replace: bool = False, gpu: str = "0"):
    if not fiftyone_replace(dataset_name, replace):
        return False
    if config.get("host") == "fiftyone_zoo":
        dataset = fiftyone_download(dataset_name, config, gpu=gpu)
        if dataset is False:
            return False
        dataset.save()
    if config.get("host") == "roboflow":
        if not roboflow_download(dataset_name, config, gpu=gpu):
            return False
    if config.get("host") == "apple_dms":
        if not apple_dms_download(dataset_name, config):
            return False
        if config.get("label_map"):
            dataset = fiftyone.load_dataset(dataset_name)
            dataset.map_labels("ground_truth", config["label_map"]).save()
    if config.get("host") == "lvis":
        dataset = lvis_download(dataset_name, config, gpu=gpu)
        if dataset is False:
            return False
        dataset.save()
    return True

