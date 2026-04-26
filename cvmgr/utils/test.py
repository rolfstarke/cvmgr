from xml.parsers.expat import model
import fiftyone
from ultralytics import YOLO
import torch
import os

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob


from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer



os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def test():
    dataset = fiftyone.zoo.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    max_samples=500,
    shuffle=True,
    persistent=True,
)
    dataset.save()

    dataset.export(
        export_dir=str("/home/rolf/GIT/cvmgr/datasets/null"),
        dataset_type=fiftyone.types.YOLOv5Dataset,
        # dont use overwrite, it messes with the split folders
        progress=True,
    )


def test2():
    return
def test3(replace=False):
    return
