import sys
import argparse
import pandas as pd
import torch
import lightning.pytorch as pl
import wandb
import os

# Import the existing initialization logic
from tnp.utils.experiment_utils import initialize_evaluation

def run_single_evaluation(run_path, checkpoint, config_path, model_name):
    """
    Runs the evaluation logic for a single model by manipulating sys.argv
    to mimic a command line call.
    """
    print(f"\n{'='*20}\nEvaluating: {model_name}\nRun: {run_path}\nCheckpoint: {checkpoint}\n{'='*20}")

    # 1. Save the original sys.argv
    original_argv = sys.argv[:]

    # 2. Mock the command line arguments for initialize_evaluation
    # We DO NOT pass model_name here, as the external library won't recognize it.
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
        gen_test = experiment.generators.test

        lit_model.eval()

        # Store number of parameters
        num_params = sum(p.numel() for p in lit_model.parameters())

        # 4. Run the Test Loop
        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            logger=False, 
            enable_checkpointing=False # Disable saving new checkpoints
        )
        
        # Suppress progress bar for cleaner batch output if desired
        trainer.test(model=lit_model, dataloaders=gen_test)

        # 5. Extract Results
        test_result = {
            k: [result[k] for result in lit_model.test_outputs]
            for k in lit_model.test_outputs[0].keys()
        }
        
        loglik = torch.stack(test_result["loglik"])
        mean_loglik = loglik.mean().item()
        std_loglik = (loglik.std() / (len(loglik) ** 0.5)).item()

        # Handle Ground Truth LogLik if present
        mean_gt_loglik = None
        std_gt_loglik = None
        
        if "gt_loglik" in test_result:
            gt_loglik = torch.stack(test_result["gt_loglik"])
            mean_gt_loglik = gt_loglik.mean().item()
            std_gt_loglik = (gt_loglik.std() / (len(gt_loglik) ** 0.5)).item()

        # 6. Return Data Row (Added model_name)
        return {
            "model_name": model_name,
            "run_path": run_path,
            "checkpoint": checkpoint,
            "num_params": num_params,
            "mean_loglik": mean_loglik,
            "std_loglik": std_loglik,
            "mean_gt_loglik": mean_gt_loglik,
            "std_gt_loglik": std_gt_loglik
        }

    except Exception as e:
        print(f"!! Error evaluating {model_name} ({run_path}): {e}")
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
    parser = argparse.ArgumentParser(description="Batch Evaluation Script")
    
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
        default="batch_eval_results.csv", 
        help="Path to save the output CSV"
    )

    args = parser.parse_args()

    # Input Validation
    # Ensure all lists are the same length
    if not (len(args.run_paths) == len(args.checkpoints) == len(args.model_names)):
        print(f"Error: Mismatch in argument lengths.")
        print(f"Model Names: {len(args.model_names)}")
        print(f"Run Paths:   {len(args.run_paths)}")
        print(f"Checkpoints: {len(args.checkpoints)}")
        sys.exit(1)

    results = []

    # Loop through all models
    # Added model_names to the zip loop
    for run_path, checkpoint, model_name in zip(args.run_paths, args.checkpoints, args.model_names):
        result_data = run_single_evaluation(run_path, checkpoint, args.config, model_name)
        results.append(result_data)

    # Create DataFrame and Save
    df = pd.DataFrame(results)
    
    # Reorder columns for readability (Put model_name first)
    cols = ["model_name", "run_path", "checkpoint", "num_params", "mean_loglik", "std_loglik", "mean_gt_loglik", "std_gt_loglik"]
    
    # Add error column if it exists in data
    if "error" in df.columns:
        cols.append("error")
    
    # Filter columns that actually exist (avoids KeyError if logic changes)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(df)
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")

if __name__ == "__main__":
    main()