import pathlib
import logging
import yaml
logger = logging.getLogger('cvmgr')
import time
import torch
import gc
import ultralytics
import wandb

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)

resources_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml"
with resources_path.open('r') as file:
    resources_yaml = yaml.safe_load(file)

import os

def optimize_hyperp_ray(dataset_name: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = resources_yaml["optimize"]["cuda_visible_devices"]
    cfg = resources_yaml["optimize"]

    model = ultralytics.YOLO(cfg["model"])

    ultralytics.settings.update({"wandb": False})
    wandb.login(key=secrets_yaml["wandb"]["api_key"])

    start_time = time.time()
    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"

    result_grid = model.tune(
        data=dataset_yaml,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        search_alg=cfg["search_alg"],
        use_ray=True,
        gpu_per_trial=cfg["gpu_per_trial"],
        iterations=cfg["iterations"],
    )

    # Log best hyperparameters to WandB as a versioned artifact
    best = result_grid.get_best_result(metric="metrics/mAP50-95(M)", mode="max")
    best_params = {k: v for k, v in best.config.items() if k != "data"}
    best_params["mAP50-95_M"] = best.metrics.get("metrics/mAP50-95(M)")

    with wandb.init(
        project=secrets_yaml["wandb"].get("project", "cvmgr"),
        name=f"tune_{dataset_name}",
        job_type="hyperparameter-tuning",
        tags=[dataset_name],
        config=best_params,
    ) as run:
        artifact = wandb.Artifact(
            name=f"{dataset_name}_best_hyperparameters",
            type="hyperparameters",
            description=f"Best hyperparameters from Ray Tune for {dataset_name}",
            metadata=best_params,
        )
        run.log_artifact(artifact)
        run.summary["mAP50-95_M"] = best_params["mAP50-95_M"]
    logger.info(f"Best hyperparameters logged to WandB artifact: {dataset_name}_best_hyperparameters | mAP50-95(M): {best_params['mAP50-95_M']:.4f}")

    # Write YOLO-compatible cfg YAML for use in training via cfg=
    _yolo_train_keys = {
        "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum",
        "box", "cls", "cls_pw", "dfl",
        "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective",
        "flipud", "fliplr", "bgr",
        "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic",
    }
    configs_dir = pathlib.Path("cvmgr") / "configs" / "training"
    existing = sorted(
        configs_dir.glob(f"{dataset_name}_*.yaml"),
        key=lambda p: float(p.stem.rsplit("_", 1)[-1].replace("-", ".")),
    )
    base_cfg_path = existing[-1] if existing else configs_dir / "default.yaml"
    with base_cfg_path.open("r") as f:
        base_cfg = yaml.safe_load(f) or {}
    base_cfg.update({k: v for k, v in best_params.items() if k in _yolo_train_keys})
    base_cfg["project"] = secrets_yaml["wandb"].get("project", "cvmgr")
    base_cfg["name"] = dataset_name
    mAP_score = best_params["mAP50-95_M"]
    cfg_path = configs_dir / f"{dataset_name}_{mAP_score:.2f}".replace(".", "-")
    cfg_path = cfg_path.with_suffix(".yaml")
    with cfg_path.open("w") as f:
        yaml.dump(base_cfg, f, default_flow_style=False, sort_keys=True)
    logger.info(f"YOLO training cfg written to {cfg_path}")

    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed for dataset: {dataset_name} | Time: {elapsed_time/3600:.1f}h {(elapsed_time%3600)/60:.0f}m")
    
    # Comprehensive memory cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()