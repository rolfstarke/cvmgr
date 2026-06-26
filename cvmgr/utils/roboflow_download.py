import pathlib
import fiftyone
import roboflow
import shutil
import os
import yaml
from .fiftyone_replace import fiftyone_replace
from .fiftyone_import import fiftyone_import
from .logging_check import util_log

@util_log("roboflow_download", success_text=lambda result, args, kwargs: "yaml_rewritten AND imported")
def roboflow_download(dataset_name: str, config: dict, gpu: str = "0"):


# formats include: clip, coco, coco-mmdetection, createml, darknet, multiclass, tensorflow, tfrecord, voc, yolokeras, yolov4pytorch, yolov4scaled, yolov5-obb, yolov5pytorch, yolov7pytorch, yolov8, yolov8-obb, yolov9, yolov11, yolov12, mt-yolov6, retinanet, benchmarker, paligemma, paligemma-txt, florence2-od, openai

    dataset = (roboflow.Roboflow(api_key=config["api_key"])
           .workspace(config["workspace"])
           .project(config["project"])
           .version(config["version"])
           .download(config["format"]))
    
    download_path = pathlib.Path(fiftyone.config.default_dataset_dir) / dataset_name
    images_dir = download_path / "images"
    labels_dir = download_path / "labels"

    images_dir.mkdir(parents=True,exist_ok=True)
    labels_dir.mkdir(parents=True,exist_ok=True)
    
    for split in config["download_splits"]:
        old_split_dir = pathlib.Path(dataset.location) / split
    
        if old_split_dir.exists():
            # Move images
            old_images = old_split_dir / "images"
            new_images = images_dir / split
            
            if old_images.exists() and any(old_images.iterdir()):
                if new_images.exists():
                    shutil.rmtree(new_images)
                shutil.move(str(old_images), str(new_images))
        
            # Move labels
            old_labels = old_split_dir / "labels"
            new_labels = labels_dir / split
            
            if old_labels.exists() and any(old_labels.iterdir()):
                if new_labels.exists():
                    shutil.rmtree(new_labels)
                shutil.move(str(old_labels), str(new_labels))
    
    # Update YAML file - read from original, save as new file
    original_yaml_path = pathlib.Path(dataset.location) / "data.yaml"  # Original file
    new_yaml_path = download_path / "dataset.yaml"       # New file in reformatted dir
    
    if original_yaml_path.exists():
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Update paths to FiftyOne format  
        data.update({
            'train': './images/train',
            'valid': './images/valid', 
            'test': './images/test'
        })
        
        # Write to NEW data.yaml in reformatted directory
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    shutil.rmtree(pathlib.Path(dataset.location))

    return bool(fiftyone_import(dataset_name, config, gpu=gpu))



