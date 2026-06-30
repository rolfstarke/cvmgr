from ultralytics import YOLO, settings
import pathlib
import shutil
import time
import yaml
import torch
import gc
import wandb
from .logging_check import util_log

_secrets = yaml.safe_load((pathlib.Path.cwd() / "cvmgr" / "configs" / "secrets.yaml").read_text())
_resources = yaml.safe_load((pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml").read_text())


def _metric_from_results(results_obj, key: str) -> float:
    if isinstance(results_obj, dict):
        value = results_obj.get(key, 0.0)
        return float(value) if value is not None else 0.0

    results_dict = getattr(results_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        value = results_dict.get(key, 0.0)
        return float(value) if value is not None else 0.0

    return 0.0


@util_log("train_yolo_model", success_text=lambda result, args, kwargs: "trained")
def train_yolo_model(dataset_name: str, gpu: str = "0", source: str = "ray"):
    device = [int(g) for g in gpu.split(",")] if "," in gpu else int(gpu)

    wandb.login(key=_secrets["wandb"]["api_key"])
    settings.update({"wandb": True})

    configs_dir = pathlib.Path.cwd() / "cvmgr" / "configs" / "training" / source
    def _version(p):
        try:
            return float(p.stem.rsplit("_", 1)[-1].replace("-", "."))
        except ValueError:
            return -1

    existing = sorted(
        configs_dir.glob(f"{dataset_name}_*.yaml"),
        key=_version,
    )
    resolved_cfg = existing[-1] if existing else configs_dir / "default.yaml"

    with resolved_cfg.open("r") as f:
        cfg_yaml = yaml.safe_load(f)

    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"
    resource_overrides = {k: v for k, v in _resources["train"].items() if k != "model"}
    train_kwargs = {**cfg_yaml, **resource_overrides, "data": str(dataset_yaml), "device": device, "name": dataset_name}
    train_kwargs.pop("cfg", None)

    model = YOLO(cfg_yaml["model"])
    start = time.time()
    results = model.train(**train_kwargs)

    # In DDP mode (multi-GPU), model.train() returns None in the main process because
    # training runs in a subprocess. Use best_pt.exists() as the real success check.
    best_pt = model.trainer.best
    if not best_pt.exists():
        return False

    val_mAP = _metric_from_results(results, "metrics/mAP50-95(M)")
    if not val_mAP:
        results_csv = model.trainer.save_dir / "results.csv"
        if results_csv.exists():
            lines = results_csv.read_text().strip().splitlines()
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                values = [v.strip() for v in lines[-1].split(",")]
                val_mAP = float(dict(zip(headers, values)).get("metrics/mAP50-95(M)", 0.0))

    # DDP-safe: run test val and log custom summary metrics in the main process after
    # the DDP subprocess exits. Disable wandb so YOLO.val() doesn't create a second run.
    settings.update({"wandb": False})
    test_res = YOLO(str(best_pt)).val(data=str(dataset_yaml), split="test", device=device)
    test_mAP = _metric_from_results(test_res, "metrics/mAP50-95(M)")
    runtime = f"{int((time.time() - start) // 60)}m {int((time.time() - start) % 60)}s"

    # debug.log always contains "finishing run entity/project/run_id" — parse it to
    # resume the finished wandb run and update its summary without re-running training.
    debug_log = model.trainer.save_dir / "wandb" / "debug.log"
    if debug_log.exists():
        for line in reversed(debug_log.read_text().splitlines()):
            if "finishing run " in line:
                run_path = line.split("finishing run ")[-1].strip()
                try:
                    api = wandb.Api()
                    wrun = api.run(run_path)
                    wrun.summary["mAP50-95_M_"] = val_mAP
                    wrun.summary["test/mAP50-95_M_"] = test_mAP
                    wrun.summary["runtime"] = runtime
                    wrun.summary.update()
                except Exception:
                    pass
                break

    models_dir = pathlib.Path.cwd() / "models" / source
    archive_dir = models_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        models_dir.glob(f"{dataset_name}_*.pt"),
        key=lambda p: float(p.stem.rsplit("_", 1)[-1].replace("-", ".")),
    )
    existing_best = existing[-1] if existing else None
    existing_mAP = float(existing_best.stem.rsplit("_", 1)[-1].replace("-", ".")) if existing_best else 0.0

    new_model_path = models_dir / (f"{dataset_name}_{val_mAP:.5f}".replace(".", "-") + ".pt")
    shutil.copy2(best_pt, new_model_path)

    if existing_best is not None:
        loser = existing_best if val_mAP >= existing_mAP else new_model_path
        loser.rename(archive_dir / loser.name)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return True
