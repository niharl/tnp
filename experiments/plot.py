import copy
import os
from typing import Callable, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from tnp.data.gp import ReversedGPGroundTruthPredictor
import torch
from torch import nn

import wandb
from tnp.data.base import Batch
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.np_functions import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: Union[
        nn.Module,
        Callable[..., torch.distributions.Distribution],
    ],
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-4.0, 4.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 64,
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
    plot_gt: bool = True,
    plot_reversal:  bool = False,
    plot_causal: bool = False,
    plot_error_bars: bool = False,
    outfolder: str = "fig",
):
    steps = int(points_per_dim * (x_range[1] - x_range[0]))
    print('Running the plotting code now...')
    x_plot = torch.linspace(x_range[0], x_range[1], steps).to(batches[0].xc)[
        None, :, None
    ]
    print('Num fig:', num_fig)
    for i in range(num_fig):
        batch = batches[i]
        xc = batch.xc[:1]
        yc = batch.yc[:1]
        xt = batch.xt[:1]
        yt = batch.yt[:1]

        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = yt

        plot_batch = copy.deepcopy(batch)
        if plot_causal and isinstance(plot_batch, SyntheticBatch):
            if plot_batch.gt_pred is not None and hasattr(plot_batch.gt_pred, 'reversal_point'):
                assert isinstance(plot_batch.gt_pred, ReversedGPGroundTruthPredictor)
                reversal_point = plot_batch.gt_pred.reversal_point
                mask = x_plot[0, :, 0] >= reversal_point
                plot_batch.xt = x_plot[:, mask, :]
            else:
                plot_batch.xt = x_plot
                print("Warning: plot_causal is True but batch.gt_pred does not have a reversal_point attribute. Plotting all target points.")
        else:
            plot_batch.xt = x_plot

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch)
            yt_pred_dist = pred_fn(model, batch)

        model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[..., 0].numel()

        # Make figure for plotting
        fig = plt.figure(figsize=figsize)

        # Plot context and target points
        plt.scatter(
            xc[0, :, 0].cpu(),
            yc[0, :, 0].cpu(),
            c="k",
            label="Context",
            s=30,
        )

        plt.scatter(
            xt[0, :, 0].cpu(),
            yt[0, :, 0].cpu(),
            c="r",
            label="Target",
            s=30,
        )

        if plot_error_bars:
            mean, std = yt_pred_dist.mean, yt_pred_dist.stddev
            # Plot box + whisker style error bars (Mean +/- 2 StdDev)
            plt.errorbar(
                xt[0, :, 0].cpu(),
                mean[0, :, 0].cpu(),
                yerr=2.0 * std[0, :, 0].cpu(),
                fmt='o',         # distinct markers
                ls='none',       # no line connecting them
                color="tab:blue",
                ecolor="tab:blue",
                capsize=4,       # caps on the error bars
                label="Model",
                alpha=0.6
            )
        else:
            mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev
            # Plot standard continuous line + filled uncertainty
            plt.plot(
                plot_batch.xt[0, :, 0].cpu(),
                mean[0, :, 0].cpu(),
                c="tab:blue",
                lw=3,
            )

            plt.fill_between(
                plot_batch.xt[0, :, 0].cpu(),
                mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
                mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

        title_str = f"$NC = {xc.shape[1]}$ $NT = {xt.shape[1]}$ NLL = {model_nll:.3f}"

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            if plot_gt:
                with torch.no_grad():
                    gt_mean, gt_std, _ = batch.gt_pred(
                        xc=xc,
                        yc=yc,
                        xt=plot_batch.xt,
                    )
                    _, _, gt_loglik = batch.gt_pred(
                        xc=xc,
                        yc=yc,
                        xt=xt,
                        yt=yt,
                    )
                    gt_loglik = gt_loglik[
                        :1
                    ]  # Need to do this because we cache during validation
                    gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

                # Plot ground truth
                plt.plot(
                    plot_batch.xt[0, :, 0].cpu(),
                    gt_mean[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )

                plt.plot(
                    plot_batch.xt[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )

                plt.plot(
                    plot_batch.xt[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    label="Ground truth",
                    lw=3,
                )

                title_str += f" GT NLL = {gt_nll:.3f}"

            if plot_reversal and hasattr(batch.gt_pred, 'reversal_point'):
                assert isinstance(batch.gt_pred, ReversedGPGroundTruthPredictor)
                reversal_point = batch.gt_pred.reversal_point
                plt.axvline(x=reversal_point, color='green', linestyle='--', label='Reversal Point', lw=2)

                if hasattr(batch.gt_pred, 'priming_frac'):
                    title_str += f"\nPF = {batch.gt_pred.priming_frac:.2f}"
                    pass

        plt.title(title_str, fontsize=24)
        plt.grid()

        # Set axis limits
        plt.xlim(x_range)
        plt.ylim(y_lim)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        fname = f"{outfolder}/{name}/{i:03d}"
        if wandb.run is not None and logging:
            wandb.log({fname: wandb.Image(fig)})
        elif savefig:
            if not os.path.isdir(f"{outfolder}/{name}"):
                os.makedirs(f"{outfolder}/{name}")
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

def plot_gps(
        batches: List[Batch],
        num_fig: int = 5,
        figsize: Tuple[float, float] = (8.0, 6.0),
        x_range: Tuple[float, float] = (-4.0, 4.0),
        y_lim: Tuple[float, float] = (-3.0, 3.0),
        max_visible_points: int = 1000,
        name: str = "plot",
        savefig: bool = False,
        logging: bool = True,
        ):
    
    print('Running the plotting code now...')
    
    
    for i in range(num_fig):
        
        batch = batches[i]
        xc = batch.xc[:1]
        yc = batch.yc[:1]
        xt = batch.xt[:1]
        yt = batch.yt[:1]

        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = yt

        # Calculate step size to limit the number of plotted points
        nc = xc.shape[1]
        nt = xt.shape[1]
        step_c = max(1, nc // max_visible_points)
        step_t = max(1, nt // max_visible_points)

        # Make figure for plotting
        fig = plt.figure(figsize=figsize)

        # Plot context and target points
        plt.scatter(
            xc[0, ::step_c, 0].cpu(),
            yc[0, ::step_c, 0].cpu(),
            c="k",
            label="Context",
            s=30,
        )

        plt.scatter(
            xt[0, ::step_t, 0].cpu(),
            yt[0, ::step_t, 0].cpu(),
            c="r",
            label="Target",
            s=30,
        )

        title_str = f"$NC = {xc.shape[1]}$" + f" $NT = {xt.shape[1]}$"

        plt.title(title_str, fontsize=24)
        plt.grid()

        # Set axis limits
        plt.xlim(x_range)
        plt.ylim(y_lim)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        fname = f"fig/{name}/{i:03d}"
        if wandb.run is not None and logging:
            wandb.log({fname: wandb.Image(fig)})
        elif savefig:
            if not os.path.isdir(f"fig/{name}"):
                os.makedirs(f"fig/{name}")
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.show()

        plt.close()