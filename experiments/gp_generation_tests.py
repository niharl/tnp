import os
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot
from datetime import datetime
import pandas as pd
import time
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback


def main():
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    wandb.init(project="gp-performance-tests", name=timestamp)

    experiment = initialize_experiment()
    generator = experiment.generator

    nc_vals = []
    runtimes = []
    memories = []

    total_steps = ((generator.max_nc - generator.current_nc) // generator.nc_step) + 1
    for _ in tqdm(range(total_steps), desc="Generating batches"):
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

    df_time = pd.DataFrame({
        "nc": nc_vals,
        "runtime": runtimes,
    })

    # -------- Plot 1: linear scale scatter (wandb built-in)
    runtime_plot = wandb.plot.scatter(
        table=wandb.Table(dataframe=df_time),
        x="nc",
        y="runtime",
        title="GPU Runtime vs Number of Context Points"
    )
    wandb.log({"runtimes_linear": runtime_plot})

    # -------- Plot 2: log-log with gradient
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.array(nc_vals)
    y = np.array(runtimes)
    log_x = np.log10(x)
    log_y = np.log10(y)

    # Compute slope (approx exponent)
    coeffs = np.polyfit(log_x, log_y, 1)
    slope = coeffs[0]

    # Create gradient colors based on log_x
    sc = ax.scatter(x, y, c=log_x, cmap="viridis", s=50)
    plt.colorbar(sc, ax=ax, label="log₁₀(nc)")

    # Plot fit line
    fit_y = 10 ** np.polyval(coeffs, log_x)
    ax.plot(x, fit_y, "r--", label=f"fit ~ n^{slope:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context Length (n)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Log-Log Runtime Scaling (Expected ~n³)")
    ax.legend()

    wandb.log({"runtimes_loglog": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
