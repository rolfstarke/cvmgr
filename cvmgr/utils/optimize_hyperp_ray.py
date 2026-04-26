import pathlib
import logging
import yaml
logger = logging.getLogger('cvmgr')
import time
import torch
import gc
import ultralytics
import wandb
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
with secrets_path.open('r') as file:
    secrets_yaml =  yaml.safe_load(file)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def optimize_hyperp_ray(dataset_name: str):
    # Ensure all GPUs are visible for Ray Tune

    # Initialize the YOLO model
    model = ultralytics.YOLO("yolo26n-seg.pt")

    ultralytics.settings.update({"wandb": True})
    wandb.login(key=secrets_yaml["wandb"]["api_key"])

    start_time = time.time()
    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"

    # Define a searcher for segmentation model
    searcher = HyperOptSearch(
        #metric="metrics/mAP50-95(M)",  # Use mask mAP for segmentation models
        #mode="max",
    )

    result_grid = model.tune(
        data=dataset_yaml,
        epochs=2,
        search_alg=searcher,
        use_ray=True,
        gpu_per_trial=1,
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed for dataset: {dataset_name} | Time: {elapsed_time/3600:.1f}h {(elapsed_time%3600)/60:.0f}m")
    
    # Comprehensive memory cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()