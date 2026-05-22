import os
from pathlib import Path
from .logging_check import util_log

@util_log("find_inconsistent_labels", success_check=lambda result, args, kwargs: result is not False, success_text=lambda result, args, kwargs: "labels_scanned")
def find_inconsistent_labels(dataset_path):
    """Find YOLO label files that have boxes but no/incomplete segments"""
    labels_dir = Path(dataset_path) / "labels"
    if not labels_dir.exists():
        return False
    
    inconsistent_files = []
    consistent_files = []
    
    for split in ["train", "val", "test"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        
        for label_file in split_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                continue
                
            box_count = 0
            segment_count = 0
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                # Count as box annotation
                box_count += 1
                
                # Check if it has segmentation data (more than 5 values: class + 4 bbox coords)
                if len(parts) > 5:
                    segment_count += 1
            
            if box_count != segment_count:
                inconsistent_files.append({
                    'file': str(label_file),
                    'boxes': box_count, 
                    'segments': segment_count
                })
            else:
                consistent_files.append(str(label_file))
    
    return inconsistent_files, consistent_files

@util_log("remove_inconsistent_files", success_text=lambda result, args, kwargs: "removed_pairs")
def remove_inconsistent_files(inconsistent_files, dataset_path):
    """Remove images and labels that have inconsistent annotations"""
    images_dir = Path(dataset_path) / "images"
    
    removed_count = 0
    for file_info in inconsistent_files:
        label_path = Path(file_info['file'])
        
        # Find corresponding image file
        relative_label_path = label_path.relative_to(Path(dataset_path) / "labels")
        split = relative_label_path.parent.name
        image_name = label_path.stem
        
        # Try common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = images_dir / split / f"{image_name}{ext}"
            if image_path.exists():
                image_path.unlink()
                break
        
        # Remove label file
        label_path.unlink()
        removed_count += 1
    
    return removed_count