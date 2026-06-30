import os
import subprocess
import pathlib

import yaml

from .logging_check import util_log

_sam3_script = "/home/rolf/GIT/sam3/sam3_visualprompt_fiftyone.py"
_sam3_cwd = "/home/rolf/GIT/sam3"
_resources = yaml.safe_load((pathlib.Path(__file__).parents[2] / "cvmgr" / "configs" / "resources.yaml").read_text())
_sam_cfg = _resources.get("sam", {})


@util_log(
    "sam3_visualprompt",
    success_check=lambda result, args, kwargs: result is True,
    success_text=lambda result, args, kwargs: "predicted",
)
def sam3_visualprompt(
    dataset_name: str,
    gpu: str = "0",
    prompt_field: str = "visual_prompt",
    output_field: str = "visual_prompt_predictions",
    replace: bool = True,
    query_batch_size: int = _sam_cfg.get("sam3_query_batch_size", 4),
    empty_cache_every: int = _sam_cfg.get("sam3_empty_cache_every", 1),
    confidence_threshold: float = _sam_cfg.get("sam3_visual_conf_threshold", 0.05),
    visual_similarity_threshold: float = _sam_cfg.get("sam3_visual_similarity_threshold", 0.5),
):
    conda_exe = os.environ.get("CONDA_EXE", "/home/rolf/anaconda3/bin/conda")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}

    print(f"Starting SAM3 visual-prompt inference for '{dataset_name}' on GPU {gpu}")

    cmd = [
        conda_exe, "run", "-n", "sam3", "--no-capture-output",
        "python", _sam3_script,
        "--dataset_name", dataset_name,
        "--prompt_field", prompt_field,
        "--output_field", output_field,
        "--query_batch_size", str(query_batch_size),
        "--empty_cache_every", str(empty_cache_every),
        "--confidence_threshold", str(confidence_threshold),
        "--visual_similarity_threshold", str(visual_similarity_threshold),
        "--gpu", str(gpu),
    ]
    if replace:
        cmd.append("--replace")

    result = subprocess.run(cmd, cwd=_sam3_cwd, env=env)

    if result.returncode == 0:
        print(f"SAM3 visual-prompt inference completed for '{dataset_name}'")
        return True
    else:
        print(f"SAM3 visual-prompt inference failed for '{dataset_name}' (exit code {result.returncode})")
        return False