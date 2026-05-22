import pathlib
import fiftyone
from .logging_check import util_log


@util_log("fiftyone_reimport_yolo_dataset", success_text=lambda result, args, kwargs: "ground_truth_yolo_field_exists")
def fiftyone_reimport_yolo_dataset(dataset_name: str, config: dict):
    export_path = pathlib.Path.cwd() / "datasets" / dataset_name
    if not export_path.exists():
        return False

    dataset = fiftyone.load_dataset(dataset_name)

    if "ground_truth_yolo" in dataset.get_field_schema():
        dataset.delete_sample_field("ground_truth_yolo")

    splits = [("val" if s == "valid" else s) for s in config.get("export_splits", [])]

    tmp_name = f"__tmp_{dataset_name}_yolo"
    if fiftyone.dataset_exists(tmp_name):
        fiftyone.delete_dataset(tmp_name)
    tmp = fiftyone.Dataset(tmp_name)

    for split in splits:
        if (export_path / "images" / split).exists():
            tmp.add_dir(
                dataset_dir=str(export_path),
                dataset_type=fiftyone.types.YOLOv5Dataset,
                label_field="ground_truth_yolo",
                split=split,
            )

    by_name = {
        pathlib.Path(s.filepath).name: s["ground_truth_yolo"]
        for s in tmp.iter_samples()
    }
    fiftyone.delete_dataset(tmp_name)

    for sample in dataset.iter_samples(autosave=True):
        sample["ground_truth_yolo"] = by_name.get(pathlib.Path(sample.filepath).name)

    return "ground_truth_yolo" in dataset.get_field_schema()
