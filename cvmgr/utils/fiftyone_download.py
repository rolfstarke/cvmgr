
import fiftyone
from .sam3_visual_segmentation import sam3_visual_segmentation
import fiftyone.utils.iou
import logging
logger = logging.getLogger('cvmgr')



def fiftyone_download(dataset_name: str, config: dict):

    dataset = fiftyone.zoo.load_zoo_dataset(
        name_or_url=config.get("origin"),
        splits=config.get("download_splits"),
        label_types=config.get("download_type"),
        classes=config.get("download_classes"),
        max_samples=config.get("samples_per_split"),
    )

    dataset.name = dataset_name
    dataset.persistent = True

    dataset = dataset.filter_labels(
    "ground_truth",
    fiftyone.ViewField("label").is_in(config.get('download_classes'))
    )
       
    dataset = sam3_visual_segmentation(dataset=dataset)

    if fiftyone.dataset_exists(dataset_name):
        logger.info(f"Dataset {dataset_name} with {len(dataset)} samples downloaded.")
    


    return dataset

