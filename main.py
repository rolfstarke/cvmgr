import argparse
import fiftyone
import fiftyone.utils.random
import yaml
import pathlib
from cvmgr import fiftyone_download

pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    datasets = yaml.safe_load(file)

# create an ArgumentParser object and a mutually exclusive group. add command line arguments and parse the arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--pipeline", help="use the pipeline.yaml", action='store_true')
args = parser.parse_args() 

try:
    if args.pipeline:
        for dataset in pipeline.get("datasets_to_download", []):
            dataset_config = datasets.get(dataset)
            if dataset_config.get("host") == "fiftyone_zoo":
                fiftyone_download(name=dataset, config=dataset_config)


except Exception as e:
    print(f"An error occurred: {e}")
    