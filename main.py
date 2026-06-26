import os
import signal
# os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# this needs to be importet before the utils
import pathlib
import logging



def _kill_tree(*_):
    import psutil
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()
    os.kill(os.getpid(), signal.SIGKILL)

signal.signal(signal.SIGINT, _kill_tree)
signal.signal(signal.SIGTERM, _kill_tree)

from cvmgr.utils.logging_check import configure_cvmgr_logging

configure_cvmgr_logging()
logging.getLogger("fiftyone").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.INFO)

import argparse
import fiftyone
import fiftyone.utils.random
import yaml

import sys


from cvmgr import fetch_dataset
from cvmgr import redistribute_splits
from cvmgr import export_yolo_dataset
from cvmgr import train_yolo_model
from cvmgr import sam3_visual_segmentation
from cvmgr import sam3_concept_segmentation
from cvmgr import test, test2, test3
#from cvmgr import concept_segmentation
from cvmgr import evaluate_model
from cvmgr import evaluate_model_sahi
#from cvmgr import optimize_hyperp
from cvmgr import optimize_hyperp_ray
from cvmgr import fiftyone_import
from cvmgr import fix_mixed_labels
from cvmgr import add_negatives
from cvmgr import add_testsplit
from cvmgr import fiftyone_reimport_yolo_dataset
from cvmgr import cvat_annotate, cvat_pull_corrections
from cvmgr import sam3_visualprompt


pipeline_path = pathlib.Path('pipeline.yaml')
with pipeline_path.open('r') as file:
    pipeline_yaml = yaml.safe_load(file)

datasets_path = pathlib.Path.cwd() / "cvmgr" / "configs" / "datasets.yaml"
with datasets_path.open('r') as file:
    dataset_cfgs_yaml = yaml.safe_load(file)


parser = argparse.ArgumentParser()
parser.add_argument("--download", help="download datasets", action='store_true')
parser.add_argument("--merge", help="use the pipeline.yaml to merge datasets", action='store_true')
parser.add_argument("--train", help="train models", action='store_true')
parser.add_argument("--textprompt", help="run SAM3 concept segmentation", action='store_true')
parser.add_argument("--test", help="use the test function", action='store_true')
parser.add_argument("--evaluate", help="use the pipeline.yaml to evaluate models", action='store_true')
parser.add_argument("--sahi", help="run SAHI sliced predictions", action='store_true')
parser.add_argument("--optimize", help="optimize hyperparameters", action='store_true')
parser.add_argument("--gpu", nargs='+', metavar='GPU', default=["0"], help="GPU device(s) (e.g. 0 or 0 1)")
parser.add_argument("--iterations", help="override iterations for --optimize", type=int, default=None)
parser.add_argument("--dataset", nargs='+', metavar='DATASET', default=None, help="override pipeline.yaml dataset list for the current command")
parser.add_argument("--correct", help="send datasets_to_correct to CVAT for label correction", action='store_true')
parser.add_argument("--pull", help="pull corrected labels back from CVAT", action='store_true')
parser.add_argument("--visualprompt", help="manually draw one visual prompt box in FiftyOne and predict for that sample", action='store_true')
args = parser.parse_args()
args.gpu = ",".join(args.gpu)

VISUAL_PROMPT_DATASET = "mark_lane_leuthener_21_ORIGINAL"

try:
    if args.download:
        for dataset in (args.dataset or pipeline_yaml.get("datasets_to_download", [])):
            fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True, gpu=args.gpu)
            add_negatives(dataset_name=dataset)
            redistribute_splits(dataset_name=dataset)
            add_testsplit(dataset_name=dataset)
            export_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)
            fiftyone_reimport_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset))

    if args.merge:
        for dataset in pipeline_yaml.get("datasets_to_merge", []):
            data_cfg = dataset_cfgs_yaml.get(dataset)
            fetch_dataset(dataset_name=dataset, config=data_cfg, replace=True)
            #redistribute_splits(dataset_name=dataset)
            #tmp_dataset = fiftyone.load_dataset(dataset)
            #sam3_visual_segmentation(dataset=tmp_dataset, recalculate=False)
            #export_yolo_dataset(dataset_name=dataset, config=data_cfg, replace=True)

    if args.train:
        for dataset in (args.dataset or pipeline_yaml.get("datasets_to_train", [])):
            fix_mixed_labels(dataset_name=dataset)
            train_yolo_model(dataset_name=dataset, gpu=args.gpu)

    if args.evaluate:
        evaluate_model(replace=True)

    if args.sahi:
        evaluate_model_sahi(replace=True)

    if args.textprompt:
        sam3_concept_segmentation(
            datasets_to_segment=(args.dataset or pipeline_yaml.get("text_prompt_datasets", [])),
            dataset_cfgs=dataset_cfgs_yaml,
            gpu=args.gpu,
        )

    
    if args.test:
        #for dataset in pipeline_yaml.get("datasets_to_segment", []):
        #    for prompt in dataset_cfgs_yaml.get(dataset).get("classes", []):
        #        test(dataset_name=dataset, prompt=prompt)
        #fetch_dataset(dataset_name="oi_v7_custom_clean", config=dataset_cfgs_yaml.get("oi_v7_custom_clean"), replace=True)
        #redistribute_splits(dataset_name="oi_v7_custom_clean")
        export_yolo_dataset(dataset_name="oi_v7_custom_clean", config=dataset_cfgs_yaml.get("oi_v7_custom_clean"), replace=True)

    if args.optimize:
        for dataset in (args.dataset or pipeline_yaml.get("models_to_optimize", [])):
            export_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True)
            fiftyone_reimport_yolo_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset))
            optimize_hyperp_ray(dataset_name=dataset, gpu=args.gpu, iterations=args.iterations)


    if args.correct:
        for dataset in pipeline_yaml.get("datasets_to_correct", []):
            #fetch_dataset(dataset_name=dataset, config=dataset_cfgs_yaml.get(dataset), replace=True, gpu=args.gpu)

            cvat_annotate(dataset_name=dataset)

    if args.pull:
        for dataset in pipeline_yaml.get("datasets_to_correct", []):
            cvat_pull_corrections(dataset_name=dataset)

    if args.visualprompt:
        for dataset in (args.dataset or pipeline_yaml.get("visual_prompt_datasets", [])):
            sam3_visualprompt(dataset_name=dataset, gpu=args.gpu, replace=True)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 

finally:
    fiftyone.close_app()