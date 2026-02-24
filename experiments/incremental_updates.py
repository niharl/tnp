import torch
import wandb
import lightning.pytorch as pl
from tnp.utils.experiment_utils import initialize_evaluation

def main():
    experiment = initialize_evaluation()

    lit_model = experiment.lit_model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.inc_test

    lit_model.eval()

    if torch.cuda.is_available():
        lit_model.cuda()

    # Extract the underlying PyTorch model
    model = lit_model.model
    device = next(model.parameters()).device
    
    print("\n" + "=" * 50)
    print(f"Starting Incremental Update Test: {eval_name}")
    print(f"Model Class: {type(model).__name__}")
    print("=" * 50)

    max_diff_mean = 0.0
    max_diff_scale = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(gen_test):
            # Unpack the batch
            if hasattr(batch, 'xc'):
                xc, yc, xt = batch.xc, batch.yc, batch.xt
            elif isinstance(batch, (list, tuple)):
                xc, yc, xt = batch[0], batch[1], batch[2]
            elif isinstance(batch, dict):
                xc, yc, xt = batch['xc'], batch['yc'], batch['xt']
            else:
                raise ValueError("Unknown batch format.")

            xc = xc.to(device)
            yc = yc.to(device)
            xt = xt.to(device)

            seq_len = xc.shape[1]
            
            # Reset incremental structures for the new batch
            model.init_inc_structs()
            
            idx = 0
            step_count = 0
            
            # Loop through the sequence in chunks
            while idx < seq_len:
                # First chunk is 64, subsequent chunks are 12
                chunk_size = 64 if idx == 0 else 12
                end_idx = min(idx + chunk_size, seq_len)
                
                if end_idx == idx:
                    break
                    
                # 1. Update Incremental Model
                xc_chunk = xc[:, idx:end_idx, :]
                yc_chunk = yc[:, idx:end_idx, :]
                model.update_ctx(xc_chunk, yc_chunk)
                
                dist_inc = model.query(xt)
                mean_inc = dist_inc.mean
                scale_inc = dist_inc.scale
                
                # 2. Run Standard Model on the accumulated context up to end_idx
                xc_current = xc[:, :end_idx, :]
                yc_current = yc[:, :end_idx, :]
                
                dist_standard = model(xc_current, yc_current, xt)
                mean_standard = dist_standard.mean
                scale_standard = dist_standard.scale
                
                # 3. Compare Predictions
                diff_mean = torch.abs(mean_standard - mean_inc).max().item()
                diff_scale = torch.abs(scale_standard - scale_inc).max().item()

                max_diff_mean = max(max_diff_mean, diff_mean)
                max_diff_scale = max(max_diff_scale, diff_scale)

                if diff_mean > 1e-4 or diff_scale > 1e-4:
                    print(f"*** WARNING in Batch {batch_idx}, Step {step_count} (ctx size {end_idx}): High discrepancy! ***")
                    print(f"  -> Mean Diff: {diff_mean:.6e}")
                    print(f"  -> Scale Diff: {diff_scale:.6e}")
                
                idx = end_idx
                step_count += 1

            # Print intermediate progress per batch
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx:03d} | Max Diff Mean: {max_diff_mean:.2e} | Max Diff Scale: {max_diff_scale:.2e}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("Final Incremental Test Results")
    print("-" * 50)
    print(f"Overall Max Diff in Means:  {max_diff_mean:.6e}")
    print(f"Overall Max Diff in Scales: {max_diff_scale:.6e}")
    
    if max_diff_mean < 1e-4 and max_diff_scale < 1e-4:
        print("\nSUCCESS: The incremental predictions strictly match the standard forward pass at all context steps!")
    else:
        print("\nFAILED: Significant differences found between standard and incremental caching.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()