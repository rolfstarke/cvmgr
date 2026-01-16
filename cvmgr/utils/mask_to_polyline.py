import fiftyone

def mask_to_polyline(dataset_name: str, mask_field: str = "ground_truth"):
    dataset = fiftyone.load_dataset(dataset_name)
    backup_field = f"{mask_field}_masks"
    dataset.rename_sample_field(mask_field, backup_field)
    fiftyone.utils.labels.instances_to_polylines(dataset, backup_field, mask_field, tolerance=2)
    dataset.save()