import pathlib
import fiftyone
import roboflow
import shutil
import subprocess
import os
import yaml
import logging
logger = logging.getLogger('cvmgr')
from .fiftyone_replace import fiftyone_replace

def roboflow_download(dataset_name: str, config: dict):


# formats include: clip, coco, coco-mmdetection, createml, darknet, multiclass, tensorflow, tfrecord, voc, yolokeras, yolov4pytorch, yolov4scaled, yolov5-obb, yolov5pytorch, yolov7pytorch, yolov8, yolov8-obb, yolov9, yolov11, yolov12, mt-yolov6, retinanet, benchmarker, paligemma, paligemma-txt, florence2-od, openai

    dataset = (roboflow.Roboflow(api_key=config["api_key"])
           .workspace(config["workspace"])
           .project(config["project"])
           .version(config["version"])
           .download(config["format"]))
    
    download_path=pathlib.Path.home() / "Fiftyone" / dataset_name
    images_dir = download_path / "images"
    labels_dir = download_path / "labels"

    images_dir.mkdir(parents=True,exist_ok=True)
    labels_dir.mkdir(parents=True,exist_ok=True)
    
    for split in config["download_splits"]:
        old_split_dir = pathlib.Path(dataset.location) / split
    
        if old_split_dir.exists():
            print(f"Processing {split} split...")
            
            # Move images
            old_images = old_split_dir / "images"
            new_images = images_dir / split
            
            if old_images.exists() and any(old_images.iterdir()):
                if new_images.exists():
                    shutil.rmtree(new_images)
                shutil.move(str(old_images), str(new_images))
                print(f"  Moved images: {old_images} -> {new_images}")
            else:
                print(f"  No images found in {old_images}")
        
            # Move labels
            old_labels = old_split_dir / "labels"
            new_labels = labels_dir / split
            
            if old_labels.exists() and any(old_labels.iterdir()):
                if new_labels.exists():
                    shutil.rmtree(new_labels)
                shutil.move(str(old_labels), str(new_labels))
                print(f"  Moved labels: {old_labels} -> {new_labels}")
            else:
                print(f"  No labels found in {old_labels}")
        else:
            print(f"Split {split} not found, skipping...")
    
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

    # Replace those print statements with:
    size_mb = sum(f.stat().st_size for f in download_path.rglob('*')) / (1024*1024)
    permissions = "RWX" if all([
        os.access(download_path, os.R_OK),
        os.access(download_path, os.W_OK), 
        os.access(download_path, os.X_OK)
    ]) else "Limited"

    logger.info(f"Dataset converted: {size_mb:.1f}MB, {permissions} permissions, files moved to {download_path}")

    shutil.rmtree(pathlib.Path(dataset.location))

    fifyone_import(dataset_name, config)



