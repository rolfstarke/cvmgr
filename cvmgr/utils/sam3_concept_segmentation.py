import fiftyone
import os
import torch
import gc
from .mask_to_polyline import mask_to_polyline

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def concept_segmentation(dataset_name: str, prompt: str = None, recompute_embeddings: bool = False):

    print(f"GPU memory before training {dataset_name}: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Load SAM3 model
    model = fiftyone.zoo.load_zoo_model("facebook/sam3")
    # Load dataset
    dataset = fiftyone.load_dataset(dataset_name)
    model.pooling_strategy = "max"

    # Increase batch size to utilize multiple GPUs
    if recompute_embeddings or not "sam_embeddings" in dataset.get_field_schema():
        dataset.compute_embeddings(
            model,
            embeddings_field="sam_embeddings",
            batch_size=4
        )

    model.operation = "concept_segmentation"
    model.threshold = 0.1
    model.mask_threshold = 0.1

    model.prompt =  [
    "Desk chair",
    "Conference chair",
    "Desk",
    "Filing cabinet",
    "Heater",
    "Boiler",
    "Air flow grill",
    "Elevator",
    "Ductwork",
    "Ceiling light",
    "Wall lights",
    "Emergency escape light",
    "Mirror",
    "Toilet",
    "Sink",
    "Faucet",
    "Raised access flooring",
    "Door",
    "Window frame",
    "Glas internal",
    "Glas external"
  ]

    dataset.apply_model(
        model,
        label_field="concept_segmentation",
        batch_size=4,  # Increased for 4 GPUs
        num_workers=4
    )
    dataset.save()
    #mask_to_polyline(dataset_name=dataset_name, mask_field="concept_segmentation")

    # Comprehensive memory cleanup
    del model
