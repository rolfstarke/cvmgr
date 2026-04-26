import fiftyone
import pathlib
from .sam3_visual_segmentation import sam3_visual_segmentation
from .sam2_visual_segmentation import sam2_visual_segmentation
import logging
logger = logging.getLogger('cvmgr')
from .fiftyone_replace import fiftyone_replace

def fiftyone_import(dataset_name: str, config: dict):
    print(f"🚀 Starting import for dataset: {dataset_name}")
    
    import_path=pathlib.Path.home() / "Fiftyone" / dataset_name
    if not import_path.exists():
        import_path=pathlib.Path("/data/fiftyone") / dataset_name
    if not import_path.exists():
        logger.error(f"Neither {import_path} nor {import_path} exist. Cannot import dataset {dataset_name}.")
        return
    
    if fiftyone.dataset_exists(dataset_name):
        dataset_name = dataset_name + "_autolabeled"

    #fiftyone_replace(dataset_name, True)
    
    dataset = fiftyone.Dataset(dataset_name, persistent=True)
    for split in config.get("download_splits", []):
        split_path = import_path / "images" / split
        if split_path.exists() and any(split_path.iterdir()):
            dataset.add_dir(
                dataset_dir=str(import_path),
                dataset_type=fiftyone.types.YOLOv5Dataset,
                label_type=config.get("type"),
                split=split,
                tags=split,
                seed=42,
            )
            
    if config.get("samples_per_split"):
        view = dataset.take(config.get("samples_per_split")*len(config.get("download_splits", [])))
        dataset.delete_samples(dataset.exclude(view))
    #if config.get("type") == "detections":
    #    dataset = sam2_visual_segmentation(dataset, recalculate=False)
    
    dataset.save()
    logger.info(f"FiftyOne dataset {dataset_name} imported: {len(dataset)} samples across {len(config['download_splits'])} splits")