import fiftyone
import pathlib
import shutil
from .logging_check import util_log


@util_log("export_yolo_dataset", success_text=lambda result, args, kwargs: "dataset_yaml_exists AND split_exported")
def export_yolo_dataset(dataset_name: str, config: dict, replace: bool = False):
    export_path = pathlib.Path.cwd() / "datasets" / dataset_name

    if export_path.exists():
        if not replace:
            return True
        shutil.rmtree(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    export_classes = config.get("export_classes")
    dataset = fiftyone.load_dataset(dataset_name)
    dataset.compute_metadata()
    label_field = "ground_truth"
    use_masks = len(
        dataset.filter_labels(label_field, fiftyone.ViewField("mask") != None)
        .match(fiftyone.ViewField(f"{label_field}.detections").length() > 0)
    ) > 0

    splits = [("val" if s == "valid" else s) for s in config.get("export_splits", [])]
    for split in splits:
        dataset.match_tags(split).export(
            export_dir=str(export_path),
            dataset_type=fiftyone.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=export_classes,
            use_masks=use_masks,
            tolerance=3,
            progress=True,
            #export_media="symlink",
        )

    dataset_yaml = export_path / "dataset.yaml"
    if dataset_yaml.exists():
        dataset_yaml.write_text(dataset_yaml.read_text().replace("valid:", "val:"))
        return True

    return False
