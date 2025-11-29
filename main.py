import argparse
import fiftyone
import fiftyone.utils.random
import yaml
import pathlib
from cvmgr import fetch_dataset
from cvmgr import redistribute_splits
from cvmgr import export_yolo_dataset

pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    dataset_configs = yaml.safe_load(file)

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--download", help="use the pipeline.yaml to download datasets", action='store_true')
group.add_argument("--merge", help="use the pipeline.yaml to merge datasets",  action='store_true')
args = parser.parse_args() 

try:
    if args.download:
        for dataset in pipeline.get("datasets_to_download", []):
            current_config = dataset_configs.get(dataset)
            fetch_dataset(dataset_name=dataset, config=current_config)

    if args.merge:
        for dataset in pipeline.get("datasets_to_merge", []):
            current_config = dataset_configs.get(dataset)
            #fetch_dataset(dataset_name=dataset, config=current_config)
            #redistribute_splits(dataset_name=dataset)
            export_yolo_dataset(dataset_name=dataset, config=current_config, replace=False)


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()