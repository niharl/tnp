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
import pandas as pd
import time

def main():
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/&Y")
    wandb.init(project="gp-performance-tests", name=timestamp)

    experiment = initialize_experiment()
    generator = experiment.generator

    nc_vals = []
    runtimes = []
    memories = []

    while generator.current_nc + generator.nc_step <= generator.max_nc:
        start_time = time.perf_counter()
        torch.cuda.reset_max_memory_allocated()

        # Run commands
        batch = generator.generate_batch()

        # Measure stats
        end_time = time.perf_counter()
        runtime = end_time - start_time
        max_memory = torch.cuda.max_memory_allocated()

        # Record stats
        nc_vals.append(generator.current_nc)
        runtimes.append(runtime)
        memories.append(max_memory)

        generator.increment_lengths()

    # Convert to dataframe for interactive wandb plotting
    target_points = 1000
    skip = max(1, batch.x.shape[1] // target_points)
    df_time = pd.DataFrame({
        "nc": nc_vals,
        "runtime": runtimes,
    })

    # Create interactive scatter plot
    runtime_plot = wandb.plot.scatter(
        table=wandb.Table(dataframe=df_time),
        x="nc",
        y="runtime",
        title="GPU Runtime vs Number of Context Points"
    )

    # Convert to dataframe for interactive wandb plotting
    target_points = 1000
    skip = max(1, batch.x.shape[1] // target_points)
    df_memory = pd.DataFrame({
        "nc": nc_vals,
        "memory": memories,
    })

    # Create interactive scatter plot
    memory_plot = wandb.plot.scatter(
        table=wandb.Table(dataframe=df_memory),
        x="nc",
        y="memory",
        title="GPU Max Memory vs Number of Context Points"
    )

    # Log to wandb
    wandb.log({"runtimes": runtime_plot, "memories": memory_plot})

    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()