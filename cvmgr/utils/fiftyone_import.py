import fiftyone
import pathlib
from .sam3_visual_segmentation import sam3_visual_segmentation
import logging
logger = logging.getLogger('cvmgr')
from .fiftyone_replace import fiftyone_replace

def fifyone_import(dataset_name: str, config: dict):
    download_path=pathlib.Path.home() / "Fiftyone" / dataset_name
    dataset = fiftyone.Dataset(dataset_name, persistent=True)
    for split in config.get("download_splits", []):
        split_path = download_path / "images" / split
        if split_path.exists() and any(split_path.iterdir()):
            dataset.add_dir(
                dataset_dir=str(download_path),
                dataset_type=fiftyone.types.YOLOv5Dataset,
                label_type=config.get("type"),
                split=split,
                tags=split,
            )
            
    if config.get("samples_per_split"):
        view = dataset.take(config.get("samples_per_split")*len(config.get("download_splits", [])))
        dataset.delete_samples(dataset.exclude(view))
    if config.get("type") == "detections":
        dataset = sam3_visual_segmentation(dataset, recalculate=False)
    
    dataset.save()
    logger.info(f"FiftyOne dataset {dataset_name} imported: {len(dataset)} samples across {len(config['download_splits'])} splits")