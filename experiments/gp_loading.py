import os
import glob
import random
import torch
import gpytorch
from torch.utils.data import IterableDataset
from tnp.data.synthetic import SyntheticBatch
from tnp.data.gp import GPGroundTruthPredictor, ReversedGPGroundTruthPredictor

class ChunkedGPDataset(IterableDataset):
    def __init__(self, data_dir, shuffle_files=True):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.shuffle_files = shuffle_files
        
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}")

        # Estimate size for progress bars (assuming ~1000 batches per chunk)
        self.num_batches = len(self.files) * 1000 

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Distribute files among workers (Stride based on worker ID)
        if worker_info is None:
            file_indices = range(len(self.files))
        else:
            file_indices = range(worker_info.id, len(self.files), worker_info.num_workers)
            
        my_files = [self.files[i] for i in file_indices]
        
        # Shuffle files to ensure different data per epoch (if desired)
        if self.shuffle_files:
            random.shuffle(my_files)
            
        # 2. Iterate through files and yield batches
        for fpath in my_files:
            try:
                # Load chunk (List of dicts)
                chunk = torch.load(fpath, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
                continue
                
            for batch_dict in chunk:
                yield self._reconstruct_batch(batch_dict)

    def _reconstruct_batch(self, data):
        # Restore the Ground Truth Predictor if present
        gt_pred = None
        if "gt_pred" in data:
            gt_pred = self._reconstruct_predictor(data["gt_pred"])

        return SyntheticBatch(
            x=data["x"],
            y=data["y"],
            xt=data["xt"],
            yt=data["yt"],
            xc=data["xc"],
            yc=data["yc"],
            gt_pred=gt_pred
        )

    def _reconstruct_predictor(self, info):
        pred_type = info.get("type", "GPGroundTruthPredictor")
        
        # Dynamic class loading for Kernel/Likelihood using exact names
        k_cls_name = info["kernel_class"]
        l_cls_name = info["likelihood_class"]
        
        k_cls = getattr(gpytorch.kernels, k_cls_name)
        l_cls = getattr(gpytorch.likelihoods, l_cls_name)
        
        # Instantiate and load state
        kernel = k_cls()
        likelihood = l_cls()
        
        kernel.load_state_dict(info["kernel_state"], strict=False)
        likelihood.load_state_dict(info["likelihood_state"], strict=False)
        
        base_pred = GPGroundTruthPredictor(kernel, likelihood)
        
        # Wrap if it was a Reversed Predictor
        if pred_type == "ReversedGPGroundTruthPredictor":
            return ReversedGPGroundTruthPredictor(
                base_gt_pred=base_pred,
                reversal_point=info["reversal_point"],
                context_range=info["context_range"]
            )
        
        return base_pred