import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import wandb

# Make sure these are importable from your modified tnp.data.gp
from tnp.data.gp import GPGroundTruthPredictor, ReversedGPGroundTruthPredictor
from tnp.utils.experiment_utils import initialize_experiment
from plot import plot_gps

def serialize_batch(batch, batch_idx, seed):
    """Helper to convert a Batch object into a dictionary with serializable metadata."""
    
    # 1. Basic Tensor Data
    batch_dict = {
        "x": batch.x.cpu(),
        "y": batch.y.cpu(),
        "xt": batch.xt.cpu(),
        "yt": batch.yt.cpu(),
        "xc": batch.xc.cpu(),
        "yc": batch.yc.cpu(),
        "meta": {
            "seed": seed,
            "source": "generate_gps.py",
            "batch_idx": batch_idx
        },
    }

    # 2. Ground Truth Predictor Reconstruction Info
    if batch.gt_pred is not None:
        gt_pred = batch.gt_pred
        pred_info = {}

        if isinstance(gt_pred, ReversedGPGroundTruthPredictor):
            pred_info["type"] = "ReversedGPGroundTruthPredictor"
            pred_info["reversal_point"] = gt_pred.reversal_point
            # We must save the context_range used for this specific batch
            pred_info["context_range"] = gt_pred.context_range
            
            # Extract the underlying GP for kernel params
            base = gt_pred.base_gt_pred
            pred_info["kernel_class"] = base.kernel.__class__.__name__
            pred_info["likelihood_class"] = base.likelihood.__class__.__name__
            pred_info["kernel_state"] = base.kernel.state_dict()
            pred_info["likelihood_state"] = base.likelihood.state_dict()

        elif isinstance(gt_pred, GPGroundTruthPredictor):
            pred_info["type"] = "GPGroundTruthPredictor"
            pred_info["kernel_class"] = gt_pred.kernel.__class__.__name__
            pred_info["likelihood_class"] = gt_pred.likelihood.__class__.__name__
            pred_info["kernel_state"] = gt_pred.kernel.state_dict()
            pred_info["likelihood_state"] = gt_pred.likelihood.state_dict()
        
        batch_dict["gt_pred"] = pred_info

    return batch_dict

def main():
    timestamp = datetime.now().strftime("%H%M%S_%d-%m-%Y")
    experiment = initialize_experiment()

    # --- Configuration ---
    # Total batches to generate (e.g., 1,000,000)
    # You can add this to your yaml under params, or hardcode it here
    total_batches = experiment.params.total_batches
    
    # How many batches to store in a single .pt file (e.g., 1000)
    chunk_size = 1000 
    
    # Number of CPU workers for parallel generation
    num_workers = 4 
    # ---------------------

    if experiment.misc.logging:
        wandb.init(
            project=experiment.misc.project, 
            name=f'Gen_N={total_batches}_{timestamp}'
        )

    save_dir = f"./experiments/datasets/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Override generator limits to ensure it doesn't stop early
    experiment.generator.num_batches = total_batches

    # Use DataLoader for parallelization
    # batch_size=None ensures we get the raw Batch object from the generator
    loader = DataLoader(
        experiment.generator,
        batch_size=None, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"Generating {total_batches} batches using {num_workers} workers...")
    print(f"Saving in chunks of {chunk_size} to {save_dir}")

    current_batches = []
    current_chunk = []
    chunk_count = 0
    iterator = iter(loader)

    # We use a manual loop to control the total count strictly
    for i in tqdm(range(total_batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            # Restart iterator if generator runs out (shouldn't happen if num_batches set high)
            iterator = iter(loader)
            batch = next(iterator)

        # Serialize
        batch_data = serialize_batch(batch, batch_idx=i, seed=experiment.misc.seed)
        current_chunk.append(batch_data)
        current_batches.append(batch)

        # Save Chunk
        if len(current_chunk) >= chunk_size:
            start_idx = chunk_count * chunk_size
            end_idx = start_idx + len(current_chunk)
            
            filename = f"batches_{start_idx:07d}_to_{end_idx:07d}.pt"
            save_path = os.path.join(save_dir, filename)
            
            torch.save(current_chunk, save_path)
            
            plot_gps(current_batches)

            current_chunk = [] # Reset buffer
            current_batches = []
            chunk_count += 1

    plot_gps(current_batches)

    # Save any remaining batches
    if len(current_chunk) > 0:
        start_idx = chunk_count * chunk_size
        end_idx = start_idx + len(current_chunk)
        filename = f"batches_{start_idx:07d}_to_{end_idx:07d}.pt"
        torch.save(current_chunk, os.path.join(save_dir, filename))

    print("Generation complete.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()