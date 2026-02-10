from xml.parsers.expat import model
import fiftyone
from ultralytics import YOLO
import torch
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test():
    dataset = fiftyone.load_dataset("oi_v7_custom_1")
    print (dataset)
    results = dataset.evaluate_detections(
    "concept_segmentation",
    gt_field="ground_truth_masks",
    eval_key="eval",
    compute_mAP=True,
    )
    # Get the 10 most common classes in the dataset
    counts = dataset.count_values("ground_truth_masks.detections.label")
    classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

    # Print a classification report for the top-10 classes
    results.print_report(classes=classes_top10)
    print(results.mAP())
    
def test2(replace=False):

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    dataset = fiftyone.load_dataset("mark_lane_leuthener_custom_21_hd")

    dataset.delete_sample_field("elevator")
    model = YOLO("/home/rolf/GIT/cvmgr/models/elevator/config_x/weights/best.pt")
    dataset.apply_model(model, label_field="elevator", confidence_thresh=0.3)

    dataset.save()

def test3():
        # The Dataset or DatasetView containing the samples you wish to export
    dataset = fiftyone.load_dataset("oi_v7_custom_1")
    # The directory to which to write the exported dataset
    export_dir = "/home/rolf/GIT/cvmgr/datasets/oi_test"

    # The name of the sample field containing the label that you wish to export
    # Used when exporting labeled datasets (e.g., classification or detection)
    label_field = "ground_truth"  # for example

    # The type of dataset to export
    # Any subclass of `fiftyone.types.Dataset` is supported
    dataset_type = fiftyone.types.COCODetectionDataset  # for example

    # Export the dataset
    dataset.export(
        export_dir=export_dir,
        dataset_type=dataset_type,
        label_field=label_field,
    )