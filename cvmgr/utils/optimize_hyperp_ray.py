import gc
import os
import pathlib
import time

os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "1")

import torch
import ultralytics
import wandb
import yaml

from .logging_check import util_log

secrets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml"
with secrets_path.open('r') as file:
    secrets_yaml = yaml.safe_load(file)

resources_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml"
with resources_path.open('r') as file:
    resources_yaml = yaml.safe_load(file)

_YOLO_TRAIN_KEYS = {
    "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum",
    "box", "cls", "cls_pw", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "bgr",
    "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic",
}


@util_log("optimize_hyperp_ray", success_text=lambda result, args, kwargs: "best_result AND cfg_written")
def optimize_hyperp_ray(dataset_name: str, gpu: str = "0", iterations: int = None):
    cfg = resources_yaml["optimize"]
    visible_devices = [d.strip() for d in gpu.split(",") if d.strip()]
    if not visible_devices:
        raise ValueError("--gpu is empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    gpu_per_trial = min(max(int(cfg.get("gpu_per_trial", 1)), 1), len(visible_devices))

    ultralytics.settings.update({"wandb": False})
    wandb.login(key=secrets_yaml["wandb"]["api_key"])
    project = secrets_yaml["wandb"].get("project", "cvmgr")

    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"
    start_time = time.time()

    n_iterations = iterations if iterations is not None else cfg["iterations"]
    result_grid = ultralytics.YOLO(cfg["model"]).tune(
        data=dataset_yaml,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        cache=cfg.get("cache", False),
        search_alg=cfg["search_alg"],
        use_ray=True,
        gpu_per_trial=gpu_per_trial,
        iterations=n_iterations,
        verbose=False,
    )

    runtime = time.time() - start_time
    best = result_grid.get_best_result(metric="metrics/mAP50-95(M)", mode="max")
    best_params = {k: v for k, v in best.config.items() if k != "data"}
    new_mAP = best.metrics.get("metrics/mAP50-95(M)", 0.0)

    trial_name = pathlib.Path(best.path).name
    weights_candidates = sorted(
        pathlib.Path("/tmp/ray").glob(f"*/artifacts/*/*/working_dirs/{trial_name}/runs/*/tune/weights/best.pt"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not weights_candidates:
        raise FileNotFoundError(f"No best.pt found for trial {trial_name} under /tmp/ray")
    test_results = ultralytics.YOLO(str(weights_candidates[0])).val(data=dataset_yaml, split="test")
    test_mAP = test_results.results_dict.get("metrics/mAP50-95(M)", 0.0)

    with wandb.init(project=project, name=f"tune_{dataset_name}", job_type="hyperparameter-tuning", tags=[dataset_name]) as run:
        run.log({"mAP50-95_M_": new_mAP, "test/mAP50-95_M_": test_mAP, "runtime": f"{int(runtime // 60)}m {int(runtime % 60)}s", "iterations": n_iterations})

    configs_dir = pathlib.Path("cvmgr") / "configs" / "training"
    archive_dir = configs_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    existing = sorted(configs_dir.glob(f"{dataset_name}_*.yaml"), key=lambda p: float(p.stem.rsplit("_", 1)[-1].replace("-", ".")))
    existing_best = existing[-1] if existing else None
    existing_mAP = float(existing_best.stem.rsplit("_", 1)[-1].replace("-", ".")) if existing_best else 0.0

    base_cfg_path = existing_best if existing_best else configs_dir / "default.yaml"
    with base_cfg_path.open("r") as f:
        base_cfg = yaml.safe_load(f) or {}
    base_cfg.update({k: v for k, v in best_params.items() if k in _YOLO_TRAIN_KEYS})
    base_cfg["project"] = project
    base_cfg["name"] = dataset_name

    cfg_path = (configs_dir / f"{dataset_name}_{new_mAP:.5f}".replace(".", "-")).with_suffix(".yaml")
    with cfg_path.open("w") as f:
        yaml.dump(base_cfg, f, default_flow_style=False, sort_keys=True)

    if existing_best is not None:
        (existing_best if new_mAP >= existing_mAP else cfg_path).rename(archive_dir / (existing_best if new_mAP >= existing_mAP else cfg_path).name)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    return True
