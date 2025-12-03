
from ultralytics import YOLO
import pathlib
import logging
import yaml

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "training.yaml"
with datasets_path.open('r') as file:
    training_configs = yaml.safe_load(file)

# if no single config is passed, all configs are iterated over



def train_yolo_model(dataset_name: str, config: dict = None):

    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"

    model = YOLO(config.get("model"))
    
    training_args ={}
    for k,v in config.items():
        if v and k != "model" and k != "name":
            training_args[k] = v
    training_args["data"] = str(dataset_yaml)
    training_args["project"] = str(pathlib.Path.cwd() / "models" / dataset_name)
    training_args["name"] = config.get("name")

    results = model.train(
        **training_args,
    )

    if results:
        training_time = getattr(results, 'speed', {}).get('train', 'N/A')
        map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
        logging.getLogger(__name__).info(f"training completed on: {dataset_name} | Time: {training_time}ms/img | mAP50: {map50}")
   