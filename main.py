import os
from venv import logger
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# this needs to be importet before the utils
import pathlib
import logging

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename='logs/full.log', 
    level=logging.INFO,
    format='\033[95m%(asctime)s - %(message)s\033[0m',
    datefmt='%Y-%m-%d %H:%M:%S'
)
app_logger = logging.getLogger('cvmgr')
app_handler = logging.FileHandler('logs/selective.log')
app_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
app_logger.addHandler(app_handler)
logging.getLogger("fiftyone").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import argparse
import fiftyone
import fiftyone.utils.random
import yaml
from cvmgr import fetch_dataset
from cvmgr import redistribute_splits
from cvmgr import export_yolo_dataset
from cvmgr import train_yolo_model
from cvmgr import sam3_visual_segmentation
from cvmgr import test, test2
from cvmgr import concept_segmentation
from cvmgr import mask_to_polyline


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
parser.add_argument("--download", help="use the pipeline.yaml to download datasets", action='store_true')
parser.add_argument("--merge", help="use the pipeline.yaml to merge datasets", action='store_true')
parser.add_argument("--train", help="use the pipeline.yaml to train models", action='store_true')
parser.add_argument("--concept", help="use the concept segmentation function of SAM3", action='store_true')
parser.add_argument("--test", help="use the test function", action='store_true')
args = parser.parse_args()

try:
    if args.download:
        for dataset in pipeline_yaml.get("datasets_to_download", []):
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)

    if args.merge:
        for dataset in pipeline_yaml.get("datasets_to_merge", []):
            data_cfg = dataset_cfgs_yaml.get(dataset)
            fetch_dataset(dataset_name=dataset, config=data_cfg, replace=True)
            #redistribute_splits(dataset_name=dataset)
            #tmp_dataset = fiftyone.load_dataset(dataset)
            #sam3_visual_segmentation(dataset=tmp_dataset, recalculate=False)
            #export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=True)

        """
        for dataset in pipeline_yaml.get("datasets_to_merge", []):
            training_cfg = training_cfgs_yaml.get(dataset).get("autolabel")
            #for training_cfg in training_cfgs_yaml.get(dataset).items():
            #    train_yolo_model(dataset_name=dataset, config=training_cfg)
            train_yolo_model(dataset_name=dataset, config=training_cfg)
        """       

    if args.train:
        for dataset in pipeline_yaml.get("datasets_to_download", []):
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=False)
            export_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=False)
        datasets_to_train = pipeline_yaml.get("datasets_to_train", {})
        for dataset, configs in datasets_to_train.items():
            for config_name in configs:
                train_yolo_model(dataset_name=dataset, config=training_cfgs_yaml.get(config_name))


    if args.concept:
        for dataset in pipeline_yaml.get("datasets_to_segment", []): 
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)
            concept_segmentation(dataset_name=dataset, recompute_embeddings=False)
            #redistribute_splits(dataset_name=dataset)
            #for prompt in dataset_cfgs_yaml.get(dataset).get("classes", []):
            #    try:
            #        concept_segmentation(dataset_name=dataset, prompt=prompt, recompute_embeddings=False)
            #    except Exception as e:
            #        print(f"An error occurred during concept segmentation for dataset '{dataset}' with prompt '{prompt}': {e}")
            #        import traceback
            #        traceback.print_exc()
    
    if args.test:
        #for dataset in pipeline_yaml.get("datasets_to_segment", []):
        #    for prompt in dataset_cfgs_yaml.get(dataset).get("classes", []):
        #        test(dataset_name=dataset, prompt=prompt)
        test2()
        
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()