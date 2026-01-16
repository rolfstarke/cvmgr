import fiftyone
import logging
logger = logging.getLogger('cvmgr')

def fiftyone_replace(dataset_name: str, replace: bool = False):
    if fiftyone.dataset_exists(dataset_name):
        if replace:
            fiftyone.delete_dataset(dataset_name)
        else:
            logger.info(f"Dataset {dataset_name} already exists. Skipping download/import.")
            return False
    return True