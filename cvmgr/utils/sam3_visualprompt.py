import os
import subprocess

from .logging_check import util_log

_sam3_script = "/home/rolf/GIT/sam3/sam3_visualprompt_fiftyone.py"
_sam3_cwd = "/home/rolf/GIT/sam3"


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