import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback
from datetime import datetime

def main():
    experiment = initialize_experiment()
    batch = experiment.generator.generate_batch()
    
    # prepare dict from Batch dataclass instance `batch`
    batch_dict = {
        "x": batch.x.cpu(),  # keep as tensors
        "y": batch.y.cpu(),
        "xt": batch.xt.cpu(),
        "yt": batch.yt.cpu(),
        "xc": batch.xc.cpu(),
        "yc": batch.yc.cpu(),
        "meta": {"seed": 42, "source": "generate_v1"},
    }

    # If gt_pred exists and has kernel/likelihood
    if batch.gt_pred is not None:
        gt_pred = batch.gt_pred
        batch_dict["gt_pred"] = {
            "kernel_class": gt_pred.kernel.__class__.__name__,
            "likelihood_class": gt_pred.likelihood.__class__.__name__,
            "kernel_state": gt_pred.kernel.state_dict(),
            "likelihood_state": gt_pred.likelihood.state_dict(),
        }
    
    save_dir = "./experiments/datasets"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S_%d-%m-%Y")
    filename = f"batch_{timestamp}.pt"
    save_path = os.path.join(save_dir, filename)

    torch.save(batch_dict, save_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()