# cvmgr

A Python package for training YOLO segmentation/detection models and managing multi-source datasets for indoor building inventory — detecting objects such as boilers, ceiling lights, doors, elevators, radiators, sinks, toilets, and vents in commercial/office environments.

## Setup

```bash
conda env create -f environment.yml
conda activate cvmgr
```

To update an existing environment:

```bash
conda env update --file environment.yml --prune
```

## Workflow

All utils are imported directly from `cvmgr`:

```python
import cvmgr
import yaml

datasets_cfg = yaml.safe_load(open("cvmgr/configs/datasets.yaml"))
config = datasets_cfg["door"]
```

### 1. Fetch dataset into FiftyOne

Downloads from the configured source (FiftyOne Zoo, Roboflow, Apple DMS, or LVIS) and stores persistently in the local FiftyOne database.

```python
cvmgr.fetch_dataset("door", config, replace=False, gpu="0")
```

### 2. Auto-segment with SAM3 (optional)

Generates segmentation masks for bounding-box annotations using SAM3 visual prompts:

```python
import fiftyone
dataset = fiftyone.load_dataset("door")
cvmgr.sam3_visual_segmentation(dataset, recalculate=False, gpu="0")
```

Or use text/concept prompts via a separate SAM3 conda environment:

```python
cvmgr.sam3_concept_segmentation(["door"], datasets_cfg, gpu="0")
```

### 3. Add hard negatives (optional)

Adds up to 20 % confusable negative samples (from `oi_v7_custom_negative`) to reduce false positives:

```python
cvmgr.add_negatives("door")
```

Hard negatives are configured per class in `datasets.yaml` under `oi_v7_custom_negative.hard_negatives`.

### 4. Export to YOLO format

Converts the FiftyOne dataset to YOLO polygon format under `datasets/<name>/`:

```python
cvmgr.export_yolo_dataset("door", config, replace=False)
```

Detections with masks are exported as polygon segments; bounding-box-only detections fall back to a rectangular polygon.

### 5. Tune hyperparameters (optional)

Runs Ray Tune + Optuna to find optimal YOLO training hyperparameters. Saves the best config to `cvmgr/configs/training/ray/<dataset>_<mAP>.yaml`.

```python
cvmgr.optimize_hyperp_ray("door", gpu="0", iterations=60)
```

### 6. Train

Picks the best existing config from `cvmgr/configs/training/<source>/` (or `default.yaml`), trains, runs a test-split validation, and saves the best `.pt` to `models/<source>/`.

```python
cvmgr.train_yolo_model("door", gpu="0", source="ray")
```

WandB logs val and test mAP. When a new model beats the previous best, the old one is moved to `models/<source>/archive/`.

### 7. Evaluate

Runs all models under `models/<source>/` against a FiftyOne dataset and writes predictions as a label field:

```python
cvmgr.evaluate_model(dataset_name="mark_lane_leuthener_21_ORIGINAL", source="ray", replace=False)
```

For sliced inference on large images (SAHI):

```python
cvmgr.evaluate_model_sahi(dataset_name="mark_lane_leuthener_21_ORIGINAL", source="ray")
```

## Configuration

| File | Purpose |
|------|---------|
| `cvmgr/configs/datasets.yaml` | Dataset sources, splits, classes, label maps |
| `cvmgr/configs/resources.yaml` | Hardware and training hyperparameter defaults |
| `cvmgr/configs/secrets.yaml` | API keys (WandB, Roboflow — not committed) |
| `cvmgr/configs/training/ray/` | Best Ray Tune configs per dataset |
| `cvmgr/configs/training/claude/` | Manually curated training configs |

## Dataset sources

| `host` value | Source |
|---|---|
| `fiftyone_zoo` | FiftyOne Dataset Zoo (Open Images v7, etc.) |
| `roboflow` | Roboflow workspace/project |
| `apple_dms` | Apple Data Management System |
| `lvis` | LVIS dataset |

Each dataset entry in `datasets.yaml` controls download splits, classes, export splits/classes, and optional label remapping.

## Utilities reference

| Util | Description |
|---|---|
| `fetch_dataset` | Download and store a dataset in FiftyOne |
| `redistribute_splits` | Re-split a dataset 85/15 train/val |
| `add_testsplit` | Carve out a held-out test split |
| `add_negatives` | Add hard-negative samples |
| `export_yolo_dataset` | Export FiftyOne → YOLO polygon format |
| `sam3_visual_segmentation` | Generate masks with SAM3 visual prompts |
| `sam3_concept_segmentation` | Generate masks with SAM3 text prompts |
| `sam3_visualprompt` | SAM3 visual prompt variant |
| `optimize_hyperp_ray` | Hyperparameter search with Ray Tune + Optuna |
| `train_yolo_model` | Train YOLO with best available config |
| `evaluate_model` | Run models on a FiftyOne dataset |
| `evaluate_model_sahi` | Sliced inference (SAHI) evaluation |
| `fix_mixed_labels` | Fix mixed detection/polyline label fields |
| `fiftyone_reimport_yolo_dataset` | Re-import exported YOLO labels back into FiftyOne |
| `cvat_annotate` | Push samples to CVAT for annotation |
| `cvat_pull_corrections` | Pull corrected labels back from CVAT |
| `apple_dms_download` | Download from Apple DMS |
| `lvis_download` / `lvis_filter_multi_mask` | Download and filter LVIS data |
| `analyse` / `analyse2` | Dataset analysis utilities |

All utils emit structured log output via the `@util_log` decorator to `logs/full.log` and `logs/selective.log`.

## FiftyOne

```bash
# List all datasets
fiftyone datasets list

# Delete a dataset
fiftyone datasets delete <dataset_name>
```

The default dataset storage location is set in `~/.fiftyone/config.json`.

**Known issue:** if `fiftyone datasets list` returns nothing, reinstalling fiftyone and deleting `~/.fiftyone/` and `~/fiftyone/` usually fixes it.

## tmux

Scroll mode: `Ctrl-b [` — then use arrow keys or PgDn. Press `q` to exit.
