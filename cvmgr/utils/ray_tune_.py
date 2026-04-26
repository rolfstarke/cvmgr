import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ultralytics import YOLO
import wandb
import pathlib
import logging
import torch
import gc

logger = logging.getLogger('cvmgr')

def optimize_hyperp_ray(dataset_name: str):
    """Optimize YOLO26n-seg with Ray Tune and WandB integration for models_to_optimize"""
    
    def objective(config):
        """Objective function for Ray Tune optimization"""
        wandb.init(project=f"{dataset_name}-raytune", config=config, reinit=True)
        
        try:
            # Use yolo26n-seg as requested
            model = YOLO("yolo26n-seg.pt")
            
            # Construct dataset path following existing pattern
            dataset_path = pathlib.Path("datasets") / dataset_name / "dataset.yaml"
            
            logger.info(f"Starting Ray Tune trial for {dataset_name} with config: {config}")
            
            results = model.train(
                data=str(dataset_path),
                epochs=50,
                save=False,
                verbose=False,
                wandb=True,
                **config
            )
            
            fitness = results.results_dict.get('fitness', 0)
            tune.report(fitness=fitness)
            
            logger.info(f"Trial completed with fitness: {fitness}")
            
        except Exception as e:
            logger.error(f"Trial failed for {dataset_name}: {e}")
            tune.report(fitness=0)
        finally:
            wandb.finish()
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Ultralytics default search space for YOLO26n-seg
    search_space = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "box": tune.uniform(0.02, 0.2),
        "cls": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
        "mixup": tune.uniform(0.0, 1.0),
    }
    
    logger.info(f"Starting Ray Tune optimization for {dataset_name}")
    
    if not ray.is_initialized():
        ray.init()
    
    analysis = tune.run(
        objective,
        config=search_space,
        num_samples=10,
        scheduler=ASHAScheduler(metric="fitness", mode="max"),
        resources_per_trial={"gpu": 1},
        name=f"raytune_{dataset_name}",
        local_dir="./ray_results"
    )
    
    best_trial = analysis.get_best_trial("fitness", "max")
    logger.info(f"Best config for {dataset_name}: {best_trial.config}")
    logger.info(f"Best fitness for {dataset_name}: {best_trial.last_result['fitness']}")
    
    return best_trial