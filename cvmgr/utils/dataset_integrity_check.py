import pathlib
import logging

logger = logging.getLogger('cvmgr')


def fix_mixed_labels(dataset_name: str) -> int:
    """
     Scan all YOLO label files for a dataset and:
     1) convert any detection-only lines (format: class_id cx cy w h — exactly 5 values)
         into segmentation polygon lines by expanding the bounding box into its four corners
     2) remove duplicate label lines (after whitespace normalization), preserving order.

    This resolves the Ultralytics warning:
      "Box and segment counts should be equal … mixed dataset"

    Returns the total number of detection-only lines that were converted.
    """
    labels_root = pathlib.Path.cwd() / "datasets" / dataset_name / "labels"
    if not labels_root.exists():
        logger.warning(f"fix_mixed_labels: labels dir not found: {labels_root}")
        return 0

    total_fixed = 0
    total_duplicates = 0
    for label_file in labels_root.rglob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []
        file_fixed = 0
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                # Detection-only line: class_id cx cy w h
                cls, cx, cy, w, h = parts
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                x1, y1 = cx - w / 2, cy - h / 2  # top-left
                x2, y2 = cx + w / 2, cy - h / 2  # top-right
                x3, y3 = cx + w / 2, cy + h / 2  # bottom-right
                x4, y4 = cx - w / 2, cy + h / 2  # bottom-left
                new_lines.append(
                    f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"
                    f" {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                )
                file_fixed += 1
            else:
                new_lines.append(" ".join(parts) if parts else "")

        deduped_lines = []
        seen = set()
        file_duplicates = 0
        for line in new_lines:
            if not line:
                continue
            if line in seen:
                file_duplicates += 1
                continue
            seen.add(line)
            deduped_lines.append(line)

        if file_fixed or file_duplicates:
            label_file.write_text("\n".join(deduped_lines) + "\n")
            total_fixed += file_fixed
            total_duplicates += file_duplicates

    if total_fixed:
        logger.info(
            f"fix_mixed_labels: converted {total_fixed} "
            f"detection-only lines to segments in '{dataset_name}'"
        )
    else:
        logger.info(
            f"fix_mixed_labels: no mixed lines found in '{dataset_name}'"
        )

    if total_duplicates:
        logger.info(
            f"fix_mixed_labels: removed {total_duplicates} "
            f"duplicate label lines in '{dataset_name}'"
        )
    else:
        logger.info(
            f"fix_mixed_labels: no duplicate label lines found in '{dataset_name}'"
        )

    return total_fixed
