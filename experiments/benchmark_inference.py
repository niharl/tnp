import sys
import os
import argparse
import time
import pandas as pd
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

# Disable WandB entirely for speed
os.environ["WANDB_MODE"] = "disabled"

try:
    from tnp.utils.experiment_utils import initialize_evaluation
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from tnp.utils.experiment_utils import initialize_evaluation

def run_fast_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/configs/benchmarks/inference_benchmark_fixed_exact.yml")
    parser.add_argument("--output_csv", type=str, default="experiments/eval_results/inference_timing_fast.csv")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    models = conf.models
    nc_range = list(range(conf.context_target.start, 
                          conf.context_target.end + 1, 
                          conf.context_target.step))
    
    num_batches = conf.benchmark_settings.num_batches
    fixed_nt = conf.target_points
    
    results = []
    
    print("="*60)
    print(f"FAST INFERENCE BENCHMARK (Single Process)")
    print("="*60)

    for model_cfg in models:
        print(f"\nLoading Model: {model_cfg.name}...")
        
        # 1. Fake sys.argv to allow initialize_evaluation to parse args
        sys.argv = [
            sys.argv[0],
            "--run_path", model_cfg.run_path,
            "--checkpoint", model_cfg.checkpoint,
            "--config", model_cfg.config
        ]
        
        # 2. Initialize Experiment ONCE per model
        experiment = initialize_evaluation()
        lit_model = experiment.lit_model
        gen_test = experiment.generators.test
        
        lit_model.eval()
        lit_model.freeze()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            lit_model = lit_model.cuda()

        # 3. Compile/Warmup ONCE per model
        # This pays the JIT compilation cost (Mamba) and CUDA init cost once.
        print(f"  Warmup (compiling kernels)...")
        dummy_batch = next(iter(gen_test))
        # Move dummy batch to device
        if torch.cuda.is_available():
            dummy_batch.xc = dummy_batch.xc.cuda()
            dummy_batch.yc = dummy_batch.yc.cuda()
            dummy_batch.xt = dummy_batch.xt.cuda()
        
        with torch.no_grad():
            lit_model(dummy_batch.xc, dummy_batch.yc, dummy_batch.xt)
        
        # 4. Loop through NC values directly
        for count in nc_range:
            print(f"  Benchmarking nc={count}...", end=" ", flush=True)
            
            # Update Generator Params dynamically
            # (Assumes generator respects these attributes or config structure)
            if hasattr(gen_test, 'min_nc'): gen_test.min_nc = count
            if hasattr(gen_test, 'max_nc'): gen_test.max_nc = count
            if hasattr(gen_test, 'min_nt'): gen_test.min_nt = fixed_nt
            if hasattr(gen_test, 'max_nt'): gen_test.max_nt = fixed_nt
            if hasattr(gen_test, 'batch_size'): gen_test.batch_size = 1
            
            # --- THE TIMING LOOP ---
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                # Manually iterate to avoid Lightning Trainer overhead
                iterator = iter(gen_test)
                for _ in range(num_batches):
                    batch = next(iterator)
                    
                    # Manual device transfer (faster than Lightning for micro-benchmarks)
                    if torch.cuda.is_available():
                        batch.xc = batch.xc.cuda()
                        batch.yc = batch.yc.cuda()
                        batch.xt = batch.xt.cuda()
                    
                    # Forward pass only (Skip GT calculation)
                    lit_model(batch.xc, batch.yc, batch.xt)

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            speed = num_batches / total_time
            print(f"{speed:.2f} it/s")
            
            results.append({
                "model": model_cfg.name,
                "nc": count,
                "it_s": speed
            })

    # Save
    if results:
        df = pd.DataFrame(results)
        pivot_df = df.pivot(index='nc', columns='model', values='it_s')
        print("\n" + "="*30)
        print("FINAL RESULTS (it/s)")
        print(pivot_df.to_string(float_format="%.2f"))
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        pivot_df.to_csv(args.output_csv)

if __name__ == "__main__":
    run_fast_benchmark()