import fiftyone
import pathlib
import shutil
import logging

logger = logging.getLogger('cvmgr')


def _build_confidence_filter(conf_thresholds: dict):
    expr = None
    for cls, thresh in conf_thresholds.items():
        conf_ok = (fiftyone.ViewField("confidence") == None) | (fiftyone.ViewField("confidence") >= thresh)
        cls_expr = (fiftyone.ViewField("label") == cls) & conf_ok
        expr = cls_expr if expr is None else (expr | cls_expr)
    return expr


def export_yolo_dataset(dataset_name: str, config: dict, replace: bool = False):
    export_path = pathlib.Path.cwd() / "datasets" / dataset_name

    if export_path.exists():
        if not replace:
            print(f"Export path {export_path} already exists. Skipping export.")
            return
        shutil.rmtree(export_path)
    export_path.mkdir(parents=True, exist_ok=True)

    export_classes = config.get("export_classes")
    if export_classes is None:
        logger.error("aborting! some export classes must be defined")
        return

    dataset = fiftyone.load_dataset(dataset_name)
    schema = dataset.get_field_schema()
    label_field = "ground_truth" if "ground_truth" in schema else None

    conf_thresholds = config.get("conf_tresholds", {})
    if conf_thresholds and label_field:
        expr = _build_confidence_filter(conf_thresholds)
        source = dataset.filter_labels(label_field, expr)
        source = source.match(fiftyone.ViewField(f"{label_field}.detections").length() > 0)
        print(f"Filtered view: {len(source)}/{len(dataset)} samples above confidence thresholds")
    else:
        source = dataset

    splits = [("val" if s == "valid" else s) for s in config.get("export_splits", [])]
    for split in splits:
        source.match_tags(split).export(
            export_dir=str(export_path),
            dataset_type=fiftyone.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=export_classes,
            use_masks=True,
            tolerance=2,
            progress=True,
            export_media="symlink",
        )

    dataset_yaml = export_path / "dataset.yaml"
    if dataset_yaml.exists():
        dataset_yaml.write_text(dataset_yaml.read_text().replace("valid:", "val:"))
