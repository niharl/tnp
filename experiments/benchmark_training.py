import subprocess
import re
import sys
import argparse
import pandas as pd
from omegaconf import OmegaConf, DictConfig, ListConfig

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

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmark_config.yml")
    args = parser.parse_args()

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
    batches_to_run = conf.benchmark_settings.batches_to_measure
    samples_per_epoch = batch_size * batches_to_run
    epochs = conf.benchmark_settings.epochs

    base_overrides = flatten_config(conf.loader_params)
    results = []

    print(f"Starting Benchmark...")
    print(f"Models: {list(models)}")
    print(f"Range: {conf.context_target.start}-{conf.context_target.end}")
    print(f"Batches per run: {batches_to_run}")
    print("-" * 60)

    nt = conf.loader_params.params.nt

    for model in models:
        for count in nc_range:
            
            cmd = [sys.executable, conf.experiment.script]

            # 1. Add Loader Params
            current_overrides = base_overrides.copy()
            
            # 2. Add Benchmark Overrides
            current_overrides.extend([
                f"params.nc={count}",
                f"params.nt={nt}",
                f"train_params.samples_per_epoch={samples_per_epoch}",
                f"train_params.batch_size={batch_size}",
                f"params.epochs={epochs}",
                "misc.logging=False",
                "misc.check_val_every_n_epoch=100",
                "misc.plot_interval=100",
                "misc.savefig=False"
            ])
            
            cmd.extend(current_overrides)
            
            # 3. Add Config File (Last)
            cmd.extend(["--config", f"experiments/configs/models/{model}.yml"])

            print(f"Benchmarking {model} | nc={count}...", end=" ", flush=True)

            try:
                # Capture both stdout (for custom stats) and stderr (for errors)
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print("FAILED")
                    print(f"Error output:\n{result.stderr[-500:]}")
                    continue

                # --- PARSING LOGIC ---
                speed = None
                
                # Priority 1: Parse the Custom Callback Output in STDOUT
                # Format: [Stats] Time: 12.34s (Train: 10.12s) | ...
                stdout_output = result.stdout
                stats_match = re.search(r'\(Train:\s*(\d+\.\d+)s\)', stdout_output)
                
                if stats_match:
                    train_time_sec = float(stats_match.group(1))
                    if train_time_sec > 0:
                        speed = batches_to_run / train_time_sec
                
                # Priority 2: Fallback to TQDM in STDERR if custom stats missing
                if speed is None:
                    stderr_output = result.stderr
                    it_s_match = re.findall(r'(\d+\.\d+)it/s', stderr_output)
                    s_it_match = re.findall(r'(\d+\.\d+)s/it', stderr_output)

                    if it_s_match:
                        speed = float(it_s_match[-1])
                    elif s_it_match:
                        speed = 1.0 / float(s_it_match[-1])

                if speed is not None:
                    print(f"Speed: {speed:.2f} it/s")
                    results.append({
                        "model": model,
                        "nc": count,
                        "it_s": speed
                    })
                else:
                    print("Parsed no speed.")
                    # Debug: print a snippet of stdout to see what happened
                    print(f"Debug Output Snippet: {stdout_output[-200:]}")

            except Exception as e:
                print(f"Exception: {e}")

    if results:
        df = pd.DataFrame(results)
        pivot_df = df.pivot(index='nc', columns='model', values='it_s')
        
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print(pivot_df.to_string(float_format="%.2f"))
        
        pivot_df.to_csv(conf.loader_params.misc.save_path)
        print(f"\nSaved to '{conf.loader_params.misc.save_path}'")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    run_benchmark()