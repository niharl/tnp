import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback

def main():
    experiment = initialize_experiment()
    batch = experiment.generator.generate_batch()
    
    import numpy as np

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

    print(batch_dict['x'][0][:10])
    print(batch_dict['y'][0][:10])

    torch.save(batch_dict, "batch_0001.pt")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()