import sys
import argparse
import copy
import os
import torch
import wandb
import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt

from tnp.utils.experiment_utils import initialize_evaluation
from tnp.data.synthetic import SyntheticBatch
from tnp.data.gp import ReversedGPGroundTruthPredictor

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

def plot_targets_only(
    model_pred_dist, 
    batch, 
    model_nll, 
    experiment,
    method_name
):
    """
    Custom plotting function that ONLY plots context points and target points for the model,
    but evaluates the Ground Truth densely to draw continuous lines.
    Returns a wandb.Image for synchronized logging.
    """
    figsize = (8.0, 6.0)
    x_range = getattr(experiment.misc, 'plot_x_range', (-4.0, 4.0))
    y_lim = (-3.0, 3.0)
    plot_gt = getattr(experiment.misc, 'plot_gt', False)
    plot_reversal = getattr(experiment.misc, 'plot_reversal', False)
    plot_causal = getattr(experiment.misc, 'plot_causal', False)

    fig = plt.figure(figsize=figsize)

    xc = batch.xc[:1]
    yc = batch.yc[:1]
    xt = batch.xt[:1]
    yt = batch.yt[:1]

    # Plot context and target true points
    plt.scatter(xc[0, :, 0].cpu(), yc[0, :, 0].cpu(), c="k", label="Context", s=30)
    plt.scatter(xt[0, :, 0].cpu(), yt[0, :, 0].cpu(), c="r", label="Target", s=30)

    # Plot Model Predictions as Error Bars at Target Points
    mean, std = model_pred_dist.mean, model_pred_dist.stddev
    plt.errorbar(
        xt[0, :, 0].cpu(),
        mean[0, :, 0].cpu(),
        yerr=2.0 * std[0, :, 0].cpu(),
        fmt='o', ls='none', color="tab:blue", ecolor="tab:blue",
        capsize=4, label="Model", alpha=0.6
    )

    # Add Method Name to Header
    title_str = f"{method_name} | $NC = {xc.shape[1]}$ $NT = {xt.shape[1]}$ NLL = {model_nll:.3f}"

    # Ground Truth Plotting (Continuous)
    if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None and plot_gt:
        # Create a dense grid for smooth plotting just like original plot.py
        steps = int(64 * (x_range[1] - x_range[0]))
        x_plot = torch.linspace(x_range[0], x_range[1], steps).to(xc.device)[None, :, None]

        # Apply causal mask to ground truth if necessary
        if plot_causal and hasattr(batch.gt_pred, 'reversal_point'):
            reversal_point = batch.gt_pred.reversal_point
            mask = x_plot[0, :, 0] >= reversal_point
            x_plot_gt = x_plot[:, mask, :]
        else:
            x_plot_gt = x_plot

        with torch.no_grad():
            gt_mean, gt_std, _ = batch.gt_pred(xc=xc, yc=yc, xt=x_plot_gt)
            _, _, gt_loglik = batch.gt_pred(xc=xc, yc=yc, xt=xt, yt=yt)
            gt_nll = -gt_loglik[:1].sum() / yt[..., 0].numel()

        # Plot continuous ground truth lines
        plt.plot(x_plot_gt[0, :, 0].cpu(), gt_mean[0, :].cpu(), "--", color="tab:purple", lw=3)
        plt.plot(x_plot_gt[0, :, 0].cpu(), gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(), "--", color="tab:purple", lw=3)
        plt.plot(x_plot_gt[0, :, 0].cpu(), gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(), "--", color="tab:purple", label="Ground truth", lw=3)

        title_str += f" GT NLL = {gt_nll:.3f}"

        if plot_reversal and hasattr(batch.gt_pred, 'reversal_point'):
            reversal_point = batch.gt_pred.reversal_point
            plt.axvline(x=reversal_point, color='green', linestyle='--', label='Reversal Point', lw=2)

    plt.title(title_str, fontsize=20)
    plt.grid()
    plt.xlim(x_range)
    plt.ylim(y_lim)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    # Create wandb image and close plot to free memory
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_single_incremental_model(run_path, checkpoint, config_path, model_name):
    print(f"\n{'='*50}\nPlotting Incremental Updates: {model_name}\nRun: {run_path}\nCheckpoint: {checkpoint}\n{'='*50}")

    original_argv = sys.argv[:]
    sys.argv = [
        "eval_plots.py", 
        "--run_path", run_path, 
        "--config", config_path, 
        "--checkpoint", checkpoint
    ]

    try:
        experiment = initialize_evaluation()
        lit_model = experiment.lit_model
        
        # Read the folder name dynamically from the yaml config
        inc_folder_name = getattr(experiment.misc, 'inc_folder', 'inc_folder')

        if hasattr(experiment.generators, 'inc_test'):
            gen_test = experiment.generators.inc_test
        else:
            gen_test = experiment.generators.test

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running evaluation on: {device}")
        
        lit_model = lit_model.to(device)
        lit_model.eval()
        model = lit_model.model

        # Ensure small number of plots based on config (fallback to 2)
        num_plots = getattr(experiment.misc, 'num_plots', 2)
        gen_test.batch_size = 1
        gen_test.num_batches = num_plots
        batches = list(iter(gen_test))[:num_plots]

        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                # Move batch to device
                if hasattr(batch, 'xc'): batch.xc = batch.xc.to(device)
                if hasattr(batch, 'yc'): batch.yc = batch.yc.to(device)
                if hasattr(batch, 'xt'): batch.xt = batch.xt.to(device)
                if hasattr(batch, 'yt'): batch.yt = batch.yt.to(device)

                xc, yc, xt, yt = batch.xc, batch.yc, batch.xt, batch.yt
                
                # Reset incremental accumulator
                model.init_inc_structs()
                seq_len = xc.shape[1]
                idx = 0
                
                # Walk through sequence length, evaluating AND plotting at each stop
                while idx < seq_len:
                    chunk_size = 64 if idx == 0 else 12
                    end_idx = min(idx + chunk_size, seq_len)
                    
                    # 1. Update Context
                    model.update_ctx(xc[:, idx:end_idx, :], yc[:, idx:end_idx, :])
                    
                    # 2. Standard Model Evaluated purely up to end_idx
                    xc_current = xc[:, :end_idx, :]
                    yc_current = yc[:, :end_idx, :]
                    dist_standard = model(xc_current, yc_current, xt)
                    model_nll_standard = -dist_standard.log_prob(yt).sum() / yt[..., 0].numel()

                    # 3. Query the incremental model via accumulated cache
                    dist_inc = model.query(xt)
                    model_nll_inc = -dist_inc.log_prob(yt).sum() / yt[..., 0].numel()

                    print(f"Plotting Batch {batch_idx:02d} | Context Steps: {end_idx:03d}")

                    # Truncate a dummy batch for accurate scatter plotting of current context
                    batch_step = copy.deepcopy(batch)
                    batch_step.xc = xc_current
                    batch_step.yc = yc_current

                    # Generate Images (Titles changed to Forward & Update)
                    img_std = plot_targets_only(
                        dist_standard, batch_step, model_nll_standard.item(), 
                        experiment, "Forward"
                    )
                    
                    img_inc = plot_targets_only(
                        dist_inc, batch_step, model_nll_inc.item(), 
                        experiment, "Update"
                    )

                    # Log simultaneously using separate distinct names for each panel
                    if wandb.run is not None:
                        wandb.log({
                            f"{inc_folder_name}/batch_{batch_idx:02d}/nc_{end_idx:03d}_standard": img_std,
                            f"{inc_folder_name}/batch_{batch_idx:02d}/nc_{end_idx:03d}_incremental": img_inc
                        })

                    idx = end_idx

    except Exception as e:
        import traceback
        print(f"!! Error plotting {model_name} ({run_path}): {e}")
        traceback.print_exc()

    finally:
        sys.argv = original_argv
        if wandb.run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Batch Incremental Target Plotting Script")
    
    parser.add_argument("--model_names", nargs='+', required=True)
    parser.add_argument("--run_paths", nargs='+', required=True)
    parser.add_argument("--checkpoints", nargs='+', required=True)
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    if not (len(args.run_paths) == len(args.checkpoints) == len(args.model_names)):
        print("Error: Mismatch in argument lengths.")
        sys.exit(1)

    for run_path, checkpoint, model_name in zip(args.run_paths, args.checkpoints, args.model_names):
        plot_single_incremental_model(run_path, checkpoint, args.config, model_name)

    print("\n" + "="*50)
    print("FINISHED PLOTTING ALL MODELS")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()