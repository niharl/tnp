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
    # Fallback if running directly inside experiments folder
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
    Runs the inference timing for a single configuration using pre-generated data.
    """
    # 1. Separate standard args (--run_path, etc) from omegaconf overrides (params.nc=1000)
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

    # 2. Initialize the experiment (downloads artifact, loads model)
    experiment = initialize_evaluation()
    lit_model = experiment.lit_model

    # 3. Parse the overrides for the loader parameters
    loader_conf = OmegaConf.from_dotlist(overrides)

    nc = loader_conf.params.nc
    nt = loader_conf.params.nt
    gp_folder = loader_conf.misc.gp_folder
    batches_to_measure = loader_conf.benchmark_settings.batches_to_measure

    # 4. Initialize Data Loader
    dataset = ChunkedGPDataset(
        gp_folder,
        shuffle_files=False,
        nc=nc,
        nt=nt
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None, # ChunkedGPDataset handles batching internally
        num_workers=0,   # Keep at 0 to avoid multiprocess overhead during timing
        pin_memory=False,
    )

    # 5. Prepare Model Devices
    lit_model.eval()
    lit_model.freeze()
    
    # --- ADD THIS FIX ---
    # Prevent PyTorch td.Normal crash from OOD 0.0 variance predictions
    if hasattr(lit_model.model, 'likelihood') and hasattr(lit_model.model.likelihood, 'min_noise'):
        lit_model.model.likelihood.min_noise = 1e-4
    # --------------------

    if torch.cuda.is_available():
        lit_model = lit_model.cuda()

    # 6. Pre-fetch Data to Memory (Isolates timing to pure model compute, no disk I/O)
    batches = []
    data_iterator = iter(dataloader)
    
    for _ in range(batches_to_measure):
        try:
            batch = next(data_iterator)
            
            # Transfer to GPU immediately
            if torch.cuda.is_available():
                if hasattr(batch, 'xc'):
                    batch.xc = batch.xc.cuda()
                    batch.yc = batch.yc.cuda()
                    batch.xt = batch.xt.cuda()
                elif isinstance(batch, (list, tuple)):
                    batch = type(batch)(t.cuda() if hasattr(t, 'cuda') else t for t in batch)
                    
            batches.append(batch)
        except StopIteration:
            print(f"Warning: Dataset only contained {len(batches)} batches. Timing on what's available.")
            break

    actual_batches = len(batches)
    if actual_batches == 0:
        raise ValueError("Dataset is empty or could not be loaded.")

    # 7. Warmup (Run 1 batch to initialize CUDA context and JIT compile kernels)
    with torch.no_grad():
        b = batches[0]
        if hasattr(b, 'xc'):
            lit_model(b.xc, b.yc, b.xt)
        else:
            lit_model(b[0], b[1], b[2]) # Fallback if batch is a tuple

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 8. Start Timing Loop
    start_time = time.time()

    with torch.no_grad():
        for b in batches:
            if hasattr(b, 'xc'):
                lit_model(b.xc, b.yc, b.xt)
            else:
                lit_model(b[0], b[1], b[2])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    speed = actual_batches / total_time

    # Output parsed by the driver script regex
    print("-" * 40)
    print(f"[Timing] Total Time: {total_time:.4f}s")
    print(f"[Timing] Speed: {speed:.2f} it/s")
    print("-" * 40)

    # Clean up WandB run so it doesn't leak or hang
    wandb.finish()


# ==========================================
# DRIVER FUNCTION (Benchmark Loop)
# ==========================================
def run_driver():
    """
    Orchestrates the benchmark by calling this script recursively as a worker.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/benchmarks/inference_loaded_config.yml", help="Path to benchmark config file")
    args, _ = parser.parse_known_args()

    try:
        conf = OmegaConf.load(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    models = conf.models
    nc_range = range(conf.context_target.start, 
                     conf.context_target.end + 1, 
                     conf.context_target.step)
    
    batch_size = conf.benchmark_settings.batch_size
    batches_to_measure = conf.benchmark_settings.batches_to_measure

    base_overrides = flatten_config(conf.loader_params)
    results = []

    print(f"Starting Inference Benchmark (ChunkedGPDataset)...")
    print(f"Models: {[m.name for m in models]}")
    print(f"NC Range: {conf.context_target.start}-{conf.context_target.end}")
    print(f"Batches per run: {batches_to_measure}")
    print("-" * 60)

    nt = conf.loader_params.params.nt
    current_script = os.path.abspath(__file__)

    for model_cfg in models:
        for count in nc_range:
            
            # Construct command to call SELF as worker
            cmd = [sys.executable, current_script, "--worker"]

            # 1. Add Inference specific args
            cmd.extend([
                "--run_path", model_cfg.run_path,
                "--checkpoint", model_cfg.checkpoint,
                "--config", model_cfg.config
            ])

            # 2. Add Loader Params
            current_overrides = base_overrides.copy()
            
            # 3. Add Benchmark Overrides
            current_overrides.extend([
                f"params.nc={count}",
                f"params.nt={nt}",
                f"benchmark_settings.batch_size={batch_size}",
                f"benchmark_settings.batches_to_measure={batches_to_measure}",
            ])
            
            cmd.extend(current_overrides)

            print(f"Benchmarking {model_cfg.name} | nc={count}...", end=" ", flush=True)

            try:
                # Run the subprocess
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print("FAILED")
                    print(f"Error output:\n{result.stderr[-500:]}")
                    continue

                # --- PARSING LOGIC ---
                speed = None
                stdout_output = result.stdout
                
                # Priority: Parse the output printed by the timing script
                match = re.search(r'\[Timing\] Speed:\s*(\d+\.\d+)\s*it/s', stdout_output)
                
                if match:
                    speed = float(match.group(1))
                    print(f"Speed: {speed:.2f} it/s")
                    results.append({
                        "model": model_cfg.name,
                        "nc": count,
                        "it_s": speed
                    })
                else:
                    print("Parsed no speed.")
                    # Uncomment to debug parsing issues
                    # print(f"Debug Output Snippet:\n{stdout_output[-500:]}")

            except Exception as e:
                print(f"Exception: {e}")

    # --- SAVE RESULTS ---
    if results:
        df = pd.DataFrame(results)
        pivot_df = df.pivot(index='nc', columns='model', values='it_s')
        
        print("\n" + "="*30)
        print("FINAL RESULTS (it/s)")
        print("="*30)
        print(pivot_df.to_string(float_format="%.2f"))
        
        save_path = conf.loader_params.misc.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pivot_df.to_csv(save_path)
        print(f"\nSaved to '{save_path}'")
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