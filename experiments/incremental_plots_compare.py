import sys
import argparse
import copy
import torch
import wandb
import lightning.pytorch as pl

from tnp.utils.experiment_utils import initialize_evaluation
from plot import plot

class IncModelWrapper:
    """
    A simple wrapper that catches the standard forward pass arguments (xc, yc, xt)
    sent by the plotting pred_fn, but safely ignores xc and yc to purely query 
    the accumulated cache targets using the incremental update method.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, xc, yc, xt, *args, **kwargs):
        return self.model.query(xt)

def plot_single_incremental_model(run_path, checkpoint, config_path, model_name):
    """
    Runs the incremental plotting logic for a single model by manipulating sys.argv
    to mimic a command line call.
    """
    print(f"\n{'='*50}\nPlotting Incremental Updates: {model_name}\nRun: {run_path}\nCheckpoint: {checkpoint}\n{'='*50}")

    # 1. Save the original sys.argv
    original_argv = sys.argv[:]

    # 2. Mock the command line arguments
    sys.argv = [
        "eval_plots.py", 
        "--run_path", run_path, 
        "--config", config_path, 
        "--checkpoint", checkpoint
    ]

    try:
        # 3. Initialize the experiment (loads config, starts WandB, loads model)
        experiment = initialize_evaluation()
        
        lit_model = experiment.lit_model
        
        # Check for inc_test in generators, else fallback to test
        if hasattr(experiment.generators, 'inc_test'):
            gen_test = experiment.generators.inc_test
        else:
            gen_test = experiment.generators.test

        # 4. Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running evaluation on: {device}")
        
        lit_model = lit_model.to(device)
        lit_model.eval()

        model = lit_model.model

        # 5. Extract a limited number of batches for plotting
        gen_test.batch_size = 1
        num_plots = getattr(experiment.misc, 'num_plots', 1)
        gen_test.num_batches = num_plots
        batches = list(iter(gen_test))[:num_plots]

        # Move batches to device manually
        for batch in batches:
            if hasattr(batch, 'xc'): batch.xc = batch.xc.to(device)
            if hasattr(batch, 'yc'): batch.yc = batch.yc.to(device)
            if hasattr(batch, 'xt'): batch.xt = batch.xt.to(device)
            if hasattr(batch, 'yt'): batch.yt = batch.yt.to(device)

        # 6. Loop through batches and chunks to generate step-by-step plots
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                seq_len = batch.xc.shape[1]
                
                # Reset incremental cache for new batch
                model.init_inc_structs()
                
                idx = 0
                while idx < seq_len:
                    # First chunk is 64, subsequent chunks are 12
                    chunk_size = 64 if idx == 0 else 12
                    end_idx = min(idx + chunk_size, seq_len)
                    
                    if end_idx == idx:
                        break
                        
                    # --- A. Update Context Cache ---
                    xc_chunk = batch.xc[:, idx:end_idx, :]
                    yc_chunk = batch.yc[:, idx:end_idx, :]
                    model.update_ctx(xc_chunk, yc_chunk)
                    
                    # Create a truncated batch representing the current context state
                    batch_step = copy.deepcopy(batch)
                    batch_step.xc = batch.xc[:, :end_idx, :]
                    batch_step.yc = batch.yc[:, :end_idx, :]
                    
                    print(f"Plotting Batch {batch_idx:02d} | Step Context Size: {end_idx:03d}")

                    # --- B. Plot Standard Forward Pass ---
                    # Using name format: batch_XX_step_YYY/standard
                    # This ensures the plot logs to inc_folder/batch_XX_step_YYY/standard/000 for ALL models
                    plot(
                        model=model,
                        batches=[batch_step],
                        num_fig=1,
                        name=f"batch_{batch_idx:02d}_step_{end_idx:03d}/standard",
                        savefig=experiment.misc.savefig,
                        logging=experiment.misc.logging,
                        pred_fn=experiment.misc.pred_fn,
                        plot_gt=experiment.misc.plot_gt,
                        x_range=getattr(experiment.misc, 'plot_x_range', (-4.0, 4.0)),
                        plot_reversal=getattr(experiment.misc, 'plot_reversal', False),
                        outfolder="inc_folder",
                        plot_causal=getattr(experiment.misc, 'plot_causal', False),
                        plot_error_bars=getattr(experiment.misc, 'plot_error_bars', False),
                    )
                    
                    # --- C. Plot Incremental Output (using cache) ---
                    # Wrap the model so the plot function only calls query(xt)
                    inc_wrapper = IncModelWrapper(model)
                    
                    plot(
                        model=inc_wrapper,
                        batches=[batch_step],
                        num_fig=1,
                        name=f"batch_{batch_idx:02d}_step_{end_idx:03d}/incremental",
                        savefig=experiment.misc.savefig,
                        logging=experiment.misc.logging,
                        pred_fn=experiment.misc.pred_fn,
                        plot_gt=experiment.misc.plot_gt,
                        x_range=getattr(experiment.misc, 'plot_x_range', (-4.0, 4.0)),
                        plot_reversal=getattr(experiment.misc, 'plot_reversal', False),
                        outfolder="inc_folder",
                        plot_causal=getattr(experiment.misc, 'plot_causal', False),
                        plot_error_bars=getattr(experiment.misc, 'plot_error_bars', False),
                    )

                    idx = end_idx

    except Exception as e:
        print(f"!! Error plotting {model_name} ({run_path}): {e}")

    finally:
        # 7. Cleanup for the next model iteration
        sys.argv = original_argv
        if wandb.run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Batch Incremental Plotting Script")
    
    parser.add_argument(
        "--model_names",
        nargs='+',
        required=True,
        help="List of friendly names for the models (must match order of run_paths)"
    )
    parser.add_argument(
        "--run_paths", 
        nargs='+', 
        required=True, 
        help="List of WandB run paths (e.g., entity/project/run_id)"
    )
    parser.add_argument(
        "--checkpoints", 
        nargs='+', 
        required=True, 
        help="List of checkpoint artifacts. Must match the order and length of --run_paths"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the generator config file"
    )

    args = parser.parse_args()

    # Input Validation
    if not (len(args.run_paths) == len(args.checkpoints) == len(args.model_names)):
        print(f"Error: Mismatch in argument lengths.")
        print(f"Model Names: {len(args.model_names)}")
        print(f"Run Paths:   {len(args.run_paths)}")
        print(f"Checkpoints: {len(args.checkpoints)}")
        sys.exit(1)

    # Loop through all models and plot
    for run_path, checkpoint, model_name in zip(args.run_paths, args.checkpoints, args.model_names):
        plot_single_incremental_model(run_path, checkpoint, args.config, model_name)

    print("\n" + "="*50)
    print("FINISHED PLOTTING ALL MODELS")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()