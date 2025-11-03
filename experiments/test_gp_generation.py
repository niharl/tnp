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
    print(type(experiment.generator))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()