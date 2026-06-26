import pathlib
import yaml
import fiftyone
import fiftyone.utils.annotations

from .logging_check import util_log


_secrets_path = pathlib.Path(__file__).parent.parent / "configs" / "secrets.yaml"


def _cvat_credentials():
    with _secrets_path.open() as f:
        secrets = yaml.safe_load(f)
    cfg = secrets.get("cvat", {})
    if not cfg.get("username") or not cfg.get("password"):
        raise RuntimeError("CVAT credentials missing in secrets.yaml")
    return cfg


@util_log("cvat_annotate")
def cvat_annotate(dataset_name: str):
    cfg = _cvat_credentials()

    dataset = fiftyone.load_dataset(dataset_name)
    if dataset_name in dataset.list_annotation_runs():
        dataset.delete_annotation_run(dataset_name)
    results = dataset.annotate(
        dataset_name,
        backend="cvat",
        label_field="ground_truth",
        label_type="instances",
        url=cfg["url"],
        username=cfg["username"],
        password=cfg["password"],
    )
    results.save()
    print(f"Task created. Key: '{dataset_name}' — fix labels in CVAT then run: python main.py --pull")
    return dataset_name


@util_log("cvat_pull_corrections")
def cvat_pull_corrections(dataset_name: str, cleanup: bool = False):
    cfg = _cvat_credentials()

    dataset = fiftyone.load_dataset(dataset_name)
    dataset.load_annotations(
        dataset_name,
        url=cfg["url"],
        username=cfg["username"],
        password=cfg["password"],
        cleanup=cleanup,
    )
    dataset.save()
    return True
