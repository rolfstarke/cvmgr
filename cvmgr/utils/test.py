import fiftyone
from ultralytics import YOLO
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test(dataset_name: str, prompt: str):
    # Load dataset
    dataset = fiftyone.load_dataset(dataset_name)
    if prompt in dataset.get_field_schema():
        dataset.delete_sample_field(prompt)
        print(f"Cleared '{prompt}' field from dataset '{dataset_name}'")
    else:
        print(f"No '{prompt}' field found in dataset '{dataset_name}'")
    dataset.save()

def test2():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    dataset = fiftyone.load_dataset("mark_lane_leuthener_custom_21_hd")
    dataset.delete_sample_field("elevator")
    model = YOLO("/home/rolf/GIT/cvmgr/models/elevator/config_x/weights/best.pt")
    dataset.apply_model(model, label_field="elevator", confidence_thresh=0.3)

    dataset.save()