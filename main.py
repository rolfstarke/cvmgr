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
logger = logging.getLogger(__name__)



pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    all_dataset_configs = yaml.safe_load(file)



parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--download", help="use the pipeline.yaml to download datasets", action='store_true')
group.add_argument("--merge", help="use the pipeline.yaml to merge datasets",  action='store_true')
group.add_argument("--train", help="use the pipeline.yaml to train models ",  action='store_true')
args = parser.parse_args() 

try:
    if args.download:
        for dataset in pipeline.get("datasets_to_download", []):
            data_cfg = all_dataset_configs.get(dataset)
            fetch_dataset(dataset_name=dataset, config=data_cfg)

    if args.merge:
        for dataset in pipeline.get("datasets_to_merge", []):
            data_cfg = all_dataset_configs.get(dataset)
            training_cfg = all_training_configs.get(dataset)
            #fetch_dataset(dataset_name=dataset, config=data_cfg)
            #redistribute_splits(dataset_name=dataset)
            #export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=False)
            train_yolo_model(dataset_name=dataset, config_name="autolabel")


    #if args.train:

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()