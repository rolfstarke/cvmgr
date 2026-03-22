from ultralytics import YOLO
import pathlib
import logging
logger = logging.getLogger('cvmgr')
import time
import torch
import gc
from ray import tune



def optimize_hyperp_ray(dataset_name: str):
    # Initialize the YOLO model
    model = YOLO("yolo26n-seg.pt")

    # Define search space - Ray Tune format from documentation
    search_space = {
        # Learning rate parameters
        "lr0": tune.uniform(1e-5, 1e-1),           # Initial learning rate
        "lrf": tune.uniform(0.01, 1.0),            # Final learning rate factor
        "momentum": tune.uniform(0.6, 0.98),       # SGD momentum factor
        "weight_decay": tune.uniform(0.0, 0.001),  # L2 regularization factor
        
        # Training schedule parameters
        "warmup_epochs": tune.uniform(0.0, 5.0),   # Number of epochs for linear learning rate warmup
        "warmup_momentum": tune.uniform(0.0, 0.95), # Initial momentum during warmup phase
        
        # Loss function weights
        "box": tune.uniform(0.02, 0.2),            # Bounding box loss weight
        "cls": tune.uniform(0.2, 4.0),             # Classification loss weight
        
        # Color augmentation parameters
        "hsv_h": tune.uniform(0.0, 0.1),           # Random hue augmentation range
        "hsv_s": tune.uniform(0.0, 0.9),           # Random saturation augmentation range
        "hsv_v": tune.uniform(0.0, 0.9),           # Random value (brightness) augmentation range
        
        # Geometric augmentation parameters
        "degrees": tune.uniform(0.0, 45.0),        # Maximum rotation augmentation in degrees
        "translate": tune.uniform(0.0, 0.9),       # Maximum translation augmentation as fraction
        "scale": tune.uniform(0.0, 0.9),           # Random scaling augmentation range
        "shear": tune.uniform(0.0, 10.0),          # Maximum shear augmentation in degrees
        "perspective": tune.uniform(0.0, 0.001),   # Random perspective augmentation range
        
        # Flip augmentation parameters
        "flipud": tune.uniform(0.0, 1.0),          # Probability of vertical image flip
        "fliplr": tune.uniform(0.0, 1.0),          # Probability of horizontal image flip
        
        # Advanced augmentation parameters
        "mosaic": tune.uniform(0.0, 1.0),          # Probability of mosaic augmentation
        "mixup": tune.uniform(0.0, 1.0),           # Probability of mixup augmentation
        "copy_paste": tune.uniform(0.0, 1.0),      # Probability of copy-paste augmentation
    }

    start_time = time.time()
    dataset_yaml = pathlib.Path.cwd() / "datasets" / dataset_name / "dataset.yaml"
    # Tune hyperparameters on COCO8 for 60 epochs
    model.tune(
        data=dataset_yaml,
        epochs=50,
        #iterations=300,
        #optimizer="AdamW",
        space=search_space,
        plots=False,
        save=False,
        val=False,
        project=str(pathlib.Path.cwd() / "models" / dataset_name),
        name="hyperparameter_optimization",
        device=[1,2,3,4],
        #gpu_per_trial=4,
        use_ray=True
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Hyperparameter optimization completed for dataset: {dataset_name} | Time: {elapsed_time/3600:.1f}h {(elapsed_time%3600)/60:.0f}m")
    
    # Comprehensive memory cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()