import argparse
import fiftyone
import fiftyone.utils.random
import yaml
import pathlib
from cvmgr import fetch_dataset
from cvmgr import redistribute_splits
from cvmgr import export_yolo_dataset
from cvmgr import train_yolo_model
import logging

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename='logs/full_pipeline.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline_yaml = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    dataset_cfgs_yaml = yaml.safe_load(file)

training_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "training.yaml"
with training_path.open('r') as file:
    training_cfgs_yaml = yaml.safe_load(file)


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--download", help="use the pipeline.yaml to download datasets", action='store_true')
group.add_argument("--merge", help="use the pipeline.yaml to merge datasets",  action='store_true')
group.add_argument("--train", help="use the pipeline.yaml to train models ",  action='store_true')
args = parser.parse_args() 

try:
    if args.download:
        for dataset in pipeline_yaml.get("datasets_to_download", []):
            data_cfg = dataset_cfgs_yaml.get(dataset)
            fetch_dataset(dataset_name=dataset, config=data_cfg)

    if args.merge:
        for dataset in pipeline_yaml.get("datasets_to_merge", []):
            data_cfg = dataset_cfgs_yaml.get(dataset)
            #fetch_dataset(dataset_name=dataset, config=data_cfg)
            #redistribute_splits(dataset_name=dataset)
            #export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=False)

            #training_cfg = training_cfgs_yaml.get(dataset).get("autolabel")
            for training_cfg in training_cfgs_yaml.get(dataset).items():
                train_yolo_model(dataset_name=dataset, config=training_cfg)


    #if args.train:

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()