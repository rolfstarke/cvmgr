import pathlib
import yaml
import roboflow
from .fiftyone_export_yolo import export_yolo_dataset
from .logging_check import util_log


@util_log("fiftyone_upload_roboflow", success_text=lambda result, args, kwargs: "exported_and_uploaded")
def fiftyone_upload_roboflow(dataset_name: str):
	datasets_cfg_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
	with datasets_cfg_path.open("r") as f:
		datasets_cfg = yaml.safe_load(f) or {}

	config = datasets_cfg.get(dataset_name)
	if not config:
		return False
	if config.get("host") != "roboflow":
		return False

	dataset_dir = pathlib.Path.cwd() / "datasets" / dataset_name
	if not dataset_dir.exists():
		if not export_yolo_dataset(dataset_name=dataset_name, config=config, replace=True):
			return False
	if not dataset_dir.exists():
		return False

	rf = roboflow.Roboflow(api_key=config["api_key"])
	project = rf.workspace(config["workspace"]).project(config["project"])
	project.upload(str(dataset_dir))
	return True