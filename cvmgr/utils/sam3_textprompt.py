import os
import json
import subprocess
import pathlib

import yaml

from .logging_check import util_log

# Multi-GPU: because asking one GPU to do everything is basically marriage.

_sam3_script = "/home/rolf/GIT/sam3/sam3_inference_fiftyone.py"
_sam3_cwd = "/home/rolf/GIT/sam3"
_resources = yaml.safe_load((pathlib.Path.cwd() / "cvmgr" / "configs" / "resources.yaml").read_text())


@util_log("sam3_textprompt")
def sam3_concept_segmentation(datasets_to_segment: list, dataset_cfgs: dict, gpu: str = "0"):
    conda_exe = os.environ.get("CONDA_EXE", "/home/rolf/anaconda3/bin/conda")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    sam_resources = _resources.get("sam", {})

    for dataset in datasets_to_segment:
        merged_config = {**sam_resources, **dataset_cfgs.get(dataset, {})}
        print(f"Starting SAM3 for {dataset} on GPUs {gpu}")
        result = subprocess.run(
            [
                conda_exe, "run", "-n", "sam3", "--no-capture-output",
                "python", _sam3_script,
                "--dataset_name", dataset,
                "--config", json.dumps(merged_config),
                "--replace",
                "--label_field", "text_prompt_predictions",
            ],
            cwd=_sam3_cwd,
            env=env,
        )
        if result.returncode == 0:
            print(f"SAM3 processing completed for {dataset}")
        else:
            print(f"SAM3 processing failed for {dataset} (exit code {result.returncode}), continuing...")
