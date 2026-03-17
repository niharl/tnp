import sys
import os
import time
import argparse
import subprocess
import re
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf, DictConfig, ListConfig

# Ensure tnp is in the python path
try:
    from tnp.utils.experiment_utils import initialize_evaluation
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tnp.utils.experiment_utils import initialize_evaluation

# Import the loader
try:
    from experiments.gp_loading import ChunkedGPDataset
except ImportError:
    from gp_loading import ChunkedGPDataset


# ==========================================
# CONFIGURATION UTILS
# ==========================================
def flatten_config(cfg, parent_key='', sep='.'):
    """Recursively flattens a nested dict config into a list of key=value strings."""
    items = []
    if isinstance(cfg, (DictConfig, dict)):
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (DictConfig, dict, ListConfig, list)):
                items.extend(flatten_config(v, new_key, sep=sep))
            else:
                items.append(f"{new_key}={v}")
    elif isinstance(cfg, (ListConfig, list)):
        for i, v in enumerate(cfg):
            new_key = f"{parent_key}[{i}]"
            if isinstance(v, (DictConfig, dict, ListConfig, list)):
                items.extend(flatten_config(v, new_key, sep=sep))
            else:
                items.append(f"{new_key}={v}")
    return items


# ==========================================
# WORKER FUNCTION (Single Run Timing)
# ==========================================
def run_worker():
    """
    Runs the incremental inference timing for a single configuration.
    """
    standard_args = []
    overrides = []
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--worker":
            i += 1
            continue
        elif arg.startswith("--"):
            if "=" in arg:
                standard_args.append(arg)
            else:
                standard_args.extend([arg, sys.argv[i+1]])
                i += 1
        elif "=" in arg:
            overrides.append(arg)
        else:
            standard_args.append(arg)
        i += 1

    sys.argv = [sys.argv[0]] + standard_args

    experiment = initialize_evaluation()
    lit_model = experiment.lit_model
    loader_conf = OmegaConf.from_dotlist(overrides)

    nc = loader_conf.params.nc
    nt = loader_conf.params.nt
    gp_folder = loader_conf.misc.gp_folder
    batches_to_measure = loader_conf.benchmark_settings.batches_to_measure

    dataset = ChunkedGPDataset(
        gp_folder,
        shuffle_files=False,
        nc=nc,
        nt=nt
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )

    lit_model.eval()
    lit_model.freeze()
    
    # Extract the underlying PyTorch model to access incremental functions
    model = lit_model.model
    
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'min_noise'):
        model.likelihood.min_noise = 1e-4

    if torch.cuda.is_available():
        lit_model = lit_model.cuda()
        model = model.cuda()

    batches = []
    data_iterator = iter(dataloader)
    
    for _ in range(batches_to_measure):
        try:
            batch = next(data_iterator)
            
            if torch.cuda.is_available():
                if hasattr(batch, 'xc'):
                    batch.xc = batch.xc.cuda()
                    batch.yc = batch.yc.cuda()
                    batch.xt = batch.xt.cuda()
                elif isinstance(batch, (list, tuple)):
                    batch = type(batch)(t.cuda() if hasattr(t, 'cuda') else t for t in batch)
                    
            batches.append(batch)
        except StopIteration:
            break

    actual_batches = len(batches)
    if actual_batches == 0:
        raise ValueError("Dataset is empty or could not be loaded.")

    # Warmup to initialise CUDA context and JIT compile kernels
    with torch.no_grad():
        b = batches[0]
        xc = b.xc if hasattr(b, 'xc') else b[0]
        yc = b.yc if hasattr(b, 'yc') else b[1]
        
        model.init_inc_structs()
        if nc > 1:
            model.update_ctx(xc[:, :nc-1, :], yc[:, :nc-1, :])
        model.update_ctx(xc[:, nc-1:nc, :], yc[:, nc-1:nc, :])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = 0.0

    # Start Timing Loop
    with torch.no_grad():
        for b in batches:
            xc = b.xc if hasattr(b, 'xc') else b[0]
            yc = b.yc if hasattr(b, 'yc') else b[1]
            
            model.init_inc_structs()
            
            # Setup phase: Pre-fill context up to nc-1 without timing it
            if nc > 1:
                model.update_ctx(xc[:, :nc-1, :], yc[:, :nc-1, :])
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # Timing phase: The incremental update for the 1 new point
            t0 = time.time()
            model.update_ctx(xc[:, nc-1:nc, :], yc[:, nc-1:nc, :])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            
            total_time += (t1 - t0)

    avg_update_time = total_time / actual_batches if actual_batches > 0 else 0
    speed = 1.0 / avg_update_time if avg_update_time > 0 else 0

    print("-" * 40)
    print(f"[Timing] Total Time for 1-point update_ctx: {total_time:.4f}s")
    print(f"[Timing] Avg Update Time: {avg_update_time:.6f} s/it")
    print(f"[Timing] Speed: {speed:.2f} it/s")
    print("-" * 40)

    wandb.finish()


# ==========================================
# DRIVER FUNCTION (Benchmark Loop)
# ==========================================
def run_driver():
    """
    Orchestrates the incremental benchmark by calling this script recursively.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/benchmarks/benchmark_update.yml", help="Path to benchmark config file")
    args, _ = parser.parse_known_args()

    try:
        conf = OmegaConf.load(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    models = conf.models
    step_size = conf.context_target.step
    nc_range = list(range(conf.context_target.start, 
                          conf.context_target.end + 1, 
                          step_size))
    
    batch_size = conf.benchmark_settings.batch_size
    batches_to_measure = conf.benchmark_settings.batches_to_measure

    base_overrides = flatten_config(conf.loader_params)
    results = []

    print("Starting Incremental Update Benchmark (1 point at a time)...")
    print(f"Models: {[m.name for m in models]}")
    print(f"NC Range: {conf.context_target.start}-{conf.context_target.end} (Checking points at intervals of {step_size})")
    print(f"Batches per run: {batches_to_measure}")
    print("-" * 60)

    nt = conf.loader_params.params.nt
    current_script = os.path.abspath(__file__)

    for model_cfg in models:
        for count in nc_range:
            
            cmd = [sys.executable, current_script, "--worker"]

            cmd.extend([
                "--run_path", model_cfg.run_path,
                "--checkpoint", model_cfg.checkpoint,
                "--config", model_cfg.config
            ])

            current_overrides = base_overrides.copy()
            
            current_overrides.extend([
                f"params.nc={count}",
                f"params.nt={nt}",
                f"benchmark_settings.batch_size={batch_size}",
                f"benchmark_settings.batches_to_measure={batches_to_measure}",
            ])
            
            cmd.extend(current_overrides)

            print(f"Benchmarking {model_cfg.name} | context size={count - 1} -> {count}...", end=" ", flush=True)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print("FAILED")
                    print(f"Error output:\n{result.stderr[-500:]}")
                    continue

                stdout_output = result.stdout
                
                match_update = re.search(r'\[Timing\] Avg Update Time:\s*(\d+\.\d+)\s*s/it', stdout_output)
                match_speed = re.search(r'\[Timing\] Speed:\s*(\d+\.\d+)\s*it/s', stdout_output)
                
                if match_update and match_speed:
                    avg_update = float(match_update.group(1))
                    speed = float(match_speed.group(1))
                    
                    print(f"Speed: {speed:.2f} it/s | Avg Update: {avg_update:.4f}s")
                    
                    results.append({
                        "model": model_cfg.name,
                        "nc": count,
                        "it_s": speed,
                        "avg_update_time_s": avg_update
                    })
                else:
                    print("Parsed no speed.")

            except Exception as e:
                print(f"Exception: {e}")

    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("FINAL RESULTS (1-point updates)")
        print("="*70)
        
        # Display Pivot Tables for better readability
        pivot_update = df.pivot(index='nc', columns='model', values='avg_update_time_s')
        
        print("\n--- Average Incremental Update Time (seconds) ---")
        print(pivot_update.to_string(float_format="%.4f"))
        
        save_path = conf.loader_params.misc.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save flat dataframe to CSV to preserve all tracked columns explicitly
        df.to_csv(save_path, index=False)
        print(f"\nSaved full results to '{save_path}'")
    else:
        print("\nNo results collected.")


# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
    else:
        run_driver()