import torch
import gpytorch
from tnp.data.gp import GPGroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch
import matplotlib.pyplot as plt
import wandb
import pandas as pd

def main():
    wandb.init(project="gp-loading-debug", name="gp_batch_scatter")


    batch_dict = torch.load("/scratches/cartwright/nl442/tnp/experiments/datasets/batch_20251108_210002.pt")

    # reconstruct tensors
    x, y, xt, yt, xc, yc = (
        batch_dict["x"],
        batch_dict["y"],
        batch_dict["xt"],
        batch_dict["yt"],
        batch_dict["xc"],
        batch_dict["yc"],
    )

    # reconstruct predictor
    if "gt_pred" in batch_dict:
        pred_info = batch_dict["gt_pred"]

        # Instantiate from class name
        kernel_cls = getattr(gpytorch.kernels, pred_info["kernel_class"])
        likelihood_cls = getattr(gpytorch.likelihoods, pred_info["likelihood_class"])

        kernel = kernel_cls()
        likelihood = likelihood_cls()

        kernel.load_state_dict(pred_info["kernel_state"])
        likelihood.load_state_dict(pred_info["likelihood_state"])

        gt_pred = GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)
    else:
        gt_pred = None

    batch = SyntheticBatch(
        x=x, y=y, xt=xt, yt=yt, xc=xc, yc=yc,
        gt_pred=gt_pred
    )

    print('First 10 x:', batch.x[0, :10])
    print('First 10 y:', batch.y[0, :10])
    print('Kernel type:', batch.gt_pred.kernel)
    print('Batch shape:', batch.x.shape)

    # Convert to dataframe for interactive wandb plotting
    df = pd.DataFrame({
        "x": batch.x[0].cpu().numpy().flatten(),
        "y": batch.y[0].cpu().numpy().flatten()
    })

    # Create interactive scatter plot
    scatter_plot = wandb.plot.scatter(
        table=wandb.Table(dataframe=df),
        x="x",
        y="y",
        title="Interactive Scatter of x vs y"
    )

    # Log to wandb
    wandb.log({"interactive_scatter_xy": scatter_plot})

    wandb.finish()

if __name__ == "__main__": 
    main()