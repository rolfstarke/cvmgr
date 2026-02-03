from xml.parsers.expat import model
import fiftyone
from ultralytics import YOLO
import torch
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test():
    model = YOLO("yolo11n.pt")
    results = model.train(data="/home/rolf/GIT/cvmgr/datasets/exitlight/dataset.yaml", epochs=10, imgsz=1280, device=[-1, -1])

def test2():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    dataset = fiftyone.load_dataset("mark_lane_leuthener_custom_21_hd")
    dataset.delete_sample_field("elevator")
    model = YOLO("/home/rolf/GIT/cvmgr/models/elevator/config_x/weights/best.pt")
    dataset.apply_model(model, label_field="elevator", confidence_thresh=0.3)

    dataset.save()