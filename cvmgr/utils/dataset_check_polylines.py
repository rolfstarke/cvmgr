import os
from pathlib import Path

def find_inconsistent_labels(dataset_path):
    """Find YOLO label files that have boxes but no/incomplete segments"""
    labels_dir = Path(dataset_path) / "labels"
    
    inconsistent_files = []
    consistent_files = []
    
    for split in ["train", "val", "test"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\nChecking {split} split...")
        
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
                print(f"Removing: {image_path}")
                image_path.unlink()
                break
        
        # Remove label file
        print(f"Removing: {label_path}")
        label_path.unlink()
        removed_count += 1
    
    return removed_count

if __name__ == "__main__":
    dataset_path = '/home/rolf/GIT/cvmgr/datasets/sink'
    
    # Find inconsistent files
    inconsistent_files, consistent_files = find_inconsistent_labels(dataset_path)
    
    print(f"\n=== RESULTS ===")
    print(f"Consistent files: {len(consistent_files)}")
    print(f"Inconsistent files: {len(inconsistent_files)}")
    
    if inconsistent_files:
        print(f"\nInconsistent files (boxes ≠ segments):")
        for file_info in inconsistent_files:
            print(f"  {file_info['file']}")
            print(f"    Boxes: {file_info['boxes']}, Segments: {file_info['segments']}")
        
        # Ask if user wants to remove them
        response = input(f"\nRemove {len(inconsistent_files)} inconsistent files? (y/n): ")
        if response.lower() == 'y':
            removed_count = remove_inconsistent_files(inconsistent_files, dataset_path)
            print(f"Removed {removed_count} inconsistent file pairs")
            print("Dataset is now consistent for segmentation training!")
        else:
            print("Files kept. You'll need to fix the annotations manually or use detection training instead.")
    else:
        print("All files are consistent! ✅")