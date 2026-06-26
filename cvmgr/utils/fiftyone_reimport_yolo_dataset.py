import pathlib
import yaml
import fiftyone
import fiftyone.utils.yolo
from .logging_check import util_log


@util_log("fiftyone_reimport_yolo_dataset", success_text=lambda result, args, kwargs: "ground_truth_yolo_field_exists")
def fiftyone_reimport_yolo_dataset(dataset_name: str, config: dict):
    export_path = pathlib.Path.cwd() / "datasets" / dataset_name
    if not export_path.exists():
        return False

    dataset = fiftyone.load_dataset(dataset_name)

    if "ground_truth_yolo" in dataset.get_field_schema():
        dataset.delete_sample_field("ground_truth_yolo")

    # Read classes from the exported dataset.yaml so the index mapping matches
    # what was actually written to disk. Fall back to config keys if missing.
    dataset_yaml_path = export_path / "dataset.yaml"
    if dataset_yaml_path.exists():
        with dataset_yaml_path.open() as f:
            dataset_yaml = yaml.safe_load(f)
        names = dataset_yaml.get("names", {})
        classes = [names[i] for i in sorted(names.keys())]
    else:
        classes = config.get("export_classes") or config.get("classes", [])

    for split in config.get("export_splits", []):
        split = "val" if split == "valid" else split
        labels_path = export_path / "labels" / split
        if not labels_path.exists():
            continue
        dataset_split = split
        fiftyone.utils.yolo.add_yolo_labels(
            dataset.match_tags(dataset_split),
            "ground_truth_yolo",
            str(labels_path),
            classes,
            label_type="polylines",
            include_missing=True,
        )

    dataset.save()

    return "ground_truth_yolo" in dataset.get_field_schema()
