from ultralytics import YOLO
import pathlib
import logging
logger = logging.getLogger('cvmgr')
import yaml
import torch
import time
import gc
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "training.yaml"
with datasets_path.open('r') as file:
    training_configs = yaml.safe_load(file)

# if no single config is passed, all configs are iterated over

def train_yolo_model(dataset_name: str, config: dict = None):

    logger.info(f"GPU memory before training {dataset_name}: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"
    model = YOLO(str(config.get("model")))

    training_args ={}
    for k,v in config.items():
        if v and k != "model":
            if isinstance(v, pathlib.Path):
                training_args[k] = str(v)
            else:
                training_args[k] = v

    training_args["data"] = str(dataset_yaml)
    training_args["project"] = str(pathlib.Path.cwd() / "models" / dataset_name)
    training_args["device"] = [-1,-1]

    print("Starting training with args:", training_args)

    start = time.time()
    results = model.train(**training_args)
    
    if results:
        elapsed = time.time() - start
        hours, minutes = int(elapsed // 3600), int((elapsed % 3600) // 60)
        map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
        logger.info(f"training completed on: {dataset_name} with {config.get('name')} | Time: {hours}h {minutes}m | mAP50: {map50}")
 
    # Comprehensive memory cleanup
    del model
    del results
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
