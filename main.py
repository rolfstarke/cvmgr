import os
from venv import logger
# os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# this needs to be importet before the utils
import pathlib
import logging

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename='logs/full.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
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

import sys
import json
import subprocess

from cvmgr import fetch_dataset
from cvmgr import redistribute_splits
from cvmgr import export_yolo_dataset
from cvmgr import train_yolo_model
from cvmgr import sam3_visual_segmentation
from cvmgr import test, test2, test3
from cvmgr import concept_segmentation
from cvmgr import mask_to_polyline
from cvmgr import evaluate_model
from cvmgr import optimize_hyperp
from cvmgr import optimize_hyperp_ray
from cvmgr import fiftyone_import

pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline_yaml = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    dataset_cfgs_yaml = yaml.safe_load(file)


parser = argparse.ArgumentParser()
parser.add_argument("--download", help="use the pipeline.yaml to download datasets", action='store_true')
parser.add_argument("--merge", help="use the pipeline.yaml to merge datasets", action='store_true')
parser.add_argument("--train", help="use the pipeline.yaml to train models", action='store_true')
parser.add_argument("--concept", help="use the concept segmentation function of SAM3", action='store_true')
parser.add_argument("--test", help="use the test function", action='store_true')
parser.add_argument("--evaluate", help="use the pipeline.yaml to evaluate models", action='store_true')
parser.add_argument("--optimize", help="use the pipeline.yaml to optimize hyperparameters", action='store_true')
args = parser.parse_args()

try:
    if args.download:
        for dataset in pipeline_yaml.get("datasets_to_download", []):
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)
            redistribute_splits(dataset_name=dataset)
            export_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)

    if args.merge:
        for dataset in pipeline_yaml.get("datasets_to_merge", []):
            data_cfg = dataset_cfgs_yaml.get(dataset)
            fetch_dataset(dataset_name=dataset, config=data_cfg, replace=True)
            #redistribute_splits(dataset_name=dataset)
            #tmp_dataset = fiftyone.load_dataset(dataset)
            #sam3_visual_segmentation(dataset=tmp_dataset, recalculate=False)
            #export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=True)

    if args.train:
        for dataset in pipeline_yaml.get("datasets_to_train", []):
            train_yolo_model(dataset_name=dataset)

    if args.evaluate:
        for model_name in pipeline_yaml.get("models_to_evaluate", []):
            models_path = pathlib.Path("/home/rolf/GIT/cvmgr/models") / model_name
            for subfolder in models_path.iterdir():
                best = models_path / subfolder / "weights" / "best.pt"
                label_field = f"{model_name}{subfolder.name}"
                try:
                    evaluate_model(prediction_model=best, prediction_labelfield=label_field, replace=True)
                except Exception as e:
                    print(f"An error occurred while evaluating model {model_name}: {e}")

    if args.concept:
        conda_exe = os.environ.get("CONDA_EXE", "/home/rolf/anaconda3/bin/conda")
        sam3_script = "/home/rolf/GIT/sam3/sam3_inference_fiftyone.py"
        for dataset in pipeline_yaml.get("datasets_to_segment", []):
            '''
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)
            
            print(f"Starting SAM3 multiclass inference for dataset: {dataset}")
            result = subprocess.run(
                [
                    conda_exe, "run", "-n", "sam3", "--no-capture-output",
                    "python", sam3_script,
                    "--dataset_name", dataset,
                    "--config", json.dumps(dataset_cfgs_yaml.get(dataset)),
                    "--replace",
                    "--label_field", "ground_truth"
                ],
                cwd="/home/rolf/GIT/sam3",
            )

            if result.returncode == 0:
                print(f"SAM3 processing completed for {dataset}")
            else:
                print(f"SAM3 processing failed for {dataset} (exit code {result.returncode}), continuing...")
            '''
            data_cfg = dataset_cfgs_yaml.get(dataset, {})
            #redistribute_splits(dataset_name=dataset)
            
            export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=True)

    
    if args.test:
        #for dataset in pipeline_yaml.get("datasets_to_segment", []):
        #    for prompt in dataset_cfgs_yaml.get(dataset).get("classes", []):
        #        test(dataset_name=dataset, prompt=prompt)


        #fetch_dataset(dataset_name="oi_v7_custom_clean", config=dataset_cfgs_yaml.get("oi_v7_custom_clean"), replace=True)
        #redistribute_splits(dataset_name="oi_v7_custom_clean")
        export_yolo_dataset(dataset_name="oi_v7_custom_clean", config=dataset_cfgs_yaml.get("oi_v7_custom_clean"), replace=True)

        #fiftyone_import(dataset_name="oi_v7_custom_clean_10", config=dataset_cfgs_yaml.get("oi_v7_custom_clean_10"))
        
        #redistribute_splits(dataset_name="oi_v7_complete")
        #export_yolo_dataset(dataset_name="oi_v7_complete", config=dataset_cfgs_yaml.get("oi_v7_complete"), replace=True)

    if args.optimize:
        #for dataset in pipeline_yaml.get("datasets_to_segment", []):
        #    for prompt in dataset_cfgs_yaml.get(dataset).get("classes", []):
        #        test(dataset_name=dataset, prompt=prompt)
        for dataset in pipeline_yaml.get("models_to_optimize", []):
            optimize_hyperp_ray(dataset_name=dataset)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()