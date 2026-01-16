import fiftyone
import pathlib
import shutil

def export_yolo_dataset(dataset_name: str, config: dict, replace: bool = False):

    export_path = pathlib.Path.cwd() / "datasets" / dataset_name

    if export_path.exists() and not replace:
        print(f"Export path {export_path} already exists. Skipping export.")
        return
    elif export_path.exists() and replace:
        shutil.rmtree(export_path)

    export_path.mkdir(parents=True, exist_ok=True)

    dataset = fiftyone.load_dataset(dataset_name)

    for split in config.get("export_splits", []):
        split_view = dataset.match_tags(split)
        split_view.export(
            export_dir=str(export_path),
            dataset_type=fiftyone.types.YOLOv5Dataset,
            label_field="ground_truth",
            # dont use overwrite, it messes with the split folders
            split=split,
            classes=config.get("export_classes"),
            progress=True,
            export_media="symlink", # "symlink" might cause issues with multigpu
        )
    # Fix dataset.yaml to use 'val' instead of 'valid'
    dataset_yaml = export_path / "dataset.yaml"
    if dataset_yaml.exists():
        content = dataset_yaml.read_text()
        dataset_yaml.write_text(content.replace("valid:", "val:"))
