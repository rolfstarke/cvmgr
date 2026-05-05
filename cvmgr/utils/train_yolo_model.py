from ultralytics import YOLO
import pathlib
import logging
logger = logging.getLogger('cvmgr')
import yaml
import torch
import time
import gc
import os
import wandb
from ultralytics import settings

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
resources_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)
with resources_path.open('r') as file:
    resources = yaml.safe_load(file)

def train_yolo_model(dataset_name: str):

    os.environ["CUDA_VISIBLE_DEVICES"] = resources["train"]["cuda_visible_devices"]

    settings.update({"wandb": True})
    wandb.login(key=secrets_yaml["wandb"]["api_key"])

    logger.info(f"GPU memory before training {dataset_name}: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    configs_dir = pathlib.Path.cwd() / "cvmgr" / "configs" / "training"
    existing = sorted(
        configs_dir.glob(f"{dataset_name}_*.yaml"),
        key=lambda p: float(p.stem.rsplit("_", 1)[-1].replace("-", ".")),
    )
    resolved_cfg = existing[-1] if existing else configs_dir / "default.yaml"
    logger.info(f"Using YOLO cfg: {resolved_cfg}")

    with resolved_cfg.open("r") as f:
        cfg_yaml = yaml.safe_load(f)
    model = YOLO(cfg_yaml.get("model"))

    start = time.time()
    train_resources = resources["train"]
    results = model.train(
        cfg=str(resolved_cfg),
        data=str(pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"),
        workers=train_resources["workers"],
        batch=train_resources["batch"],
    )

    elapsed = time.time() - start
    hours, minutes = int(elapsed // 3600), int((elapsed % 3600) // 60)
    logger.info(f"training completed on: {dataset_name} | Time: {hours}h {minutes}m")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()