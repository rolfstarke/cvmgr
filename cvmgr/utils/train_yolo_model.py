from ultralytics import YOLO
import pathlib
import yaml
import torch
import time
import gc
import os
import wandb
from ultralytics import settings
from .logging_check import util_log

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
resources_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)
with resources_path.open('r') as file:
    resources = yaml.safe_load(file)

@util_log("train_yolo_model", success_text=lambda result, args, kwargs: "train_results AND cache_reset")
def train_yolo_model(dataset_name: str):

    os.environ["CUDA_VISIBLE_DEVICES"] = resources["train"]["cuda_visible_devices"]

    settings.update({"wandb": True})
    wandb.login(key=secrets_yaml["wandb"]["api_key"])

    configs_dir = pathlib.Path.cwd() / "cvmgr" / "configs" / "training"
    existing = sorted(
        configs_dir.glob(f"{dataset_name}_*.yaml"),
        key=lambda p: float(p.stem.rsplit("_", 1)[-1].replace("-", ".")),
    )
    resolved_cfg = existing[-1] if existing else configs_dir / "default.yaml"

    with resolved_cfg.open("r") as f:
        cfg_yaml = yaml.safe_load(f)
    model = YOLO(cfg_yaml.get("model"))

    start = time.time()
    train_resources = resources["train"]
    device = train_resources.get("device", train_resources["cuda_visible_devices"])
    results = model.train(
        cfg=str(resolved_cfg),
        data=str(pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"),
        workers=train_resources["workers"],
        batch=train_resources["batch"],
        device=device,
        name=dataset_name,
    )

    elapsed = time.time() - start
    if results is None or elapsed < 0:
        return False

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    return True