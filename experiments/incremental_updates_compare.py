import sys
import argparse
import pandas as pd
import torch
import wandb
import lightning.pytorch as pl

from tnp.utils.experiment_utils import initialize_evaluation

def test_single_incremental_model(run_path, checkpoint, config_path, model_name):
    """
    Runs the incremental update test logic for a single model by manipulating sys.argv
    to mimic a command line call.
    """
    print(f"\n{'='*50}\nTesting Incremental Updates: {model_name}\nRun: {run_path}\nCheckpoint: {checkpoint}\n{'='*50}")

    # 1. Save the original sys.argv
    original_argv = sys.argv[:]

    # 2. Mock the command line arguments for initialize_evaluation
    sys.argv = [
        "eval.py", 
        "--run_path", run_path, 
        "--config", config_path, 
        "--checkpoint", checkpoint
    ]

    try:
        # 3. Initialize the experiment (loads config, model from wandb, etc.)
        experiment = initialize_evaluation()
        
        lit_model = experiment.lit_model
        
        # NOTE: Using inc_test as specified in your prompt
        gen_test = experiment.generators.inc_test

        lit_model.eval()


        if torch.cuda.is_available():
            lit_model.cuda()


        # Extract the underlying PyTorch model
        model = lit_model.model
        device = next(model.parameters()).device
        
        print(f"Model Class: {type(model).__name__}")

        max_diff_mean = 0.0
        max_diff_scale = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(gen_test):
                # Unpack the batch
                if hasattr(batch, 'xc'):
                    xc, yc, xt = batch.xc, batch.yc, batch.xt
                elif isinstance(batch, (list, tuple)):
                    xc, yc, xt = batch[0], batch[1], batch[2]
                elif isinstance(batch, dict):
                    xc, yc, xt = batch['xc'], batch['yc'], batch['xt']
                else:
                    raise ValueError("Unknown batch format.")

                xc = xc.to(device)
                yc = yc.to(device)
                xt = xt.to(device)

                seq_len = xc.shape[1]
                
                # Reset incremental structures for the new batch
                model.init_inc_structs()
                
                idx = 0
                step_count = 0
                
                # Loop through the sequence in chunks
                while idx < seq_len:
                    # First chunk is 64, subsequent chunks are 12
                    chunk_size = 64 if idx == 0 else 12
                    end_idx = min(idx + chunk_size, seq_len)
                    
                    if end_idx == idx:
                        break
                        
                    # 1. Update Incremental Model
                    xc_chunk = xc[:, idx:end_idx, :]
                    yc_chunk = yc[:, idx:end_idx, :]
                    model.update_ctx(xc_chunk, yc_chunk)
                    
                    dist_inc = model.query(xt)
                    mean_inc = dist_inc.mean
                    scale_inc = dist_inc.scale
                    
                    # 2. Run Standard Model on the accumulated context up to end_idx
                    xc_current = xc[:, :end_idx, :]
                    yc_current = yc[:, :end_idx, :]
                    
                    dist_standard = model(xc_current, yc_current, xt)
                    mean_standard = dist_standard.mean
                    scale_standard = dist_standard.scale
                    
                    # 3. Compare Predictions
                    diff_mean = torch.abs(mean_standard - mean_inc).max().item()
                    diff_scale = torch.abs(scale_standard - scale_inc).max().item()

                    max_diff_mean = max(max_diff_mean, diff_mean)
                    max_diff_scale = max(max_diff_scale, diff_scale)

                    if diff_mean > 1e-4 or diff_scale > 1e-4:
                        print(f"*** WARNING in Batch {batch_idx}, Step {step_count} (ctx size {end_idx}): High discrepancy! ***")
                        print(f"  -> Mean Diff: {diff_mean:.6e}")
                        print(f"  -> Scale Diff: {diff_scale:.6e}")
                    
                    idx = end_idx
                    step_count += 1

                # Print intermediate progress per batch
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx:03d} | Max Diff Mean: {max_diff_mean:.2e} | Max Diff Scale: {max_diff_scale:.2e}")

        # Summary for this specific model
        success = (max_diff_mean < 1e-4 and max_diff_scale < 1e-4)
        if success:
            print("\nSUCCESS: Predictions match exactly!")
        else:
            print("\nFAILED: Significant differences found.")

        # Return Data Row
        return {
            "model_name": model_name,
            "run_path": run_path,
            "checkpoint": checkpoint,
            "max_diff_mean": max_diff_mean,
            "max_diff_scale": max_diff_scale,
            "success": success
        }

    except Exception as e:
        print(f"!! Error testing {model_name} ({run_path}): {e}")
        return {
            "model_name": model_name,
            "run_path": run_path,
            "checkpoint": checkpoint,
            "error": str(e)
        }

    finally:
        # 7. Cleanup
        # Restore arguments for the next loop iteration
        sys.argv = original_argv
        
        # Ensure W&B run is closed so the next iteration can start a new one
        if wandb.run is not None:
            wandb.finish()


def main():
    # Parse arguments for the batch script
    parser = argparse.ArgumentParser(description="Batch Incremental Update Evaluation Script")
    
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
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="incremental_test_results.csv", 
        help="Path to save the output CSV"
    )

    args = parser.parse_args()

    # Input Validation
    if not (len(args.run_paths) == len(args.checkpoints) == len(args.model_names)):
        print(f"Error: Mismatch in argument lengths.")
        print(f"Model Names: {len(args.model_names)}")
        print(f"Run Paths:   {len(args.run_paths)}")
        print(f"Checkpoints: {len(args.checkpoints)}")
        sys.exit(1)

    results = []

    # Loop through all models
    for run_path, checkpoint, model_name in zip(args.run_paths, args.checkpoints, args.model_names):
        result_data = test_single_incremental_model(run_path, checkpoint, args.config, model_name)
        results.append(result_data)

    # Create DataFrame and Save
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ["model_name", "max_diff_mean", "max_diff_scale", "success", "run_path", "checkpoint", "error"]
    
    # Filter columns that actually exist (handles missing 'error' col)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n" + "="*40)
    print("FINAL INCREMENTAL RESULTS")
    print("="*40)
    print(df.to_string())
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")

if __name__ == "__main__":
    main()