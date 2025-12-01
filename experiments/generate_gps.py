import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot_gps

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback
from datetime import datetime

def main():
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    experiment = initialize_experiment()

    wandb.init(project="gp-bulk-generation-tests", name=f'N_c = {experiment.generator.min_nc}; {timestamp}')

    batches = []
    for _ in range(experiment.params.total_batches):
        batch = experiment.generator.generate_batch()
        
        # prepare dict from Batch dataclass instance `batch`
        batch_dict = {
            "x": batch.x.cpu(),  # keep as tensors
            "y": batch.y.cpu(),
            "xt": batch.xt.cpu(),
            "yt": batch.yt.cpu(),
            "xc": batch.xc.cpu(),
            "yc": batch.yc.cpu(),
            "meta": {"seed": experiment.misc.seed, 
                     "source": "generate_v1"},
        }

        batches.append(batch)

        # If gt_pred exists and has kernel/likelihood
        #if batch.gt_pred is not None:
        if False: 
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

    plot_gps(batches)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()