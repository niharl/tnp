import torch
import gpytorch
from tnp.data.gp import GPGroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch

batch_dict = torch.load("test batch.pt")

# reconstruct tensors
x, y, xt, yt, xc, yc = (
    batch_dict["x"],
    batch_dict["y"],
    batch_dict["xt"],
    batch_dict["yt"],
    batch_dict["xc"],
    batch_dict["yc"],
)

# reconstruct predictor
if "gt_pred" in batch_dict:
    pred_info = batch_dict["gt_pred"]

    # Instantiate from class name
    kernel_cls = getattr(gpytorch.kernels, pred_info["kernel_class"])
    likelihood_cls = getattr(gpytorch.likelihoods, pred_info["likelihood_class"])

    kernel = kernel_cls()
    likelihood = likelihood_cls()

    kernel.load_state_dict(pred_info["kernel_state"])
    likelihood.load_state_dict(pred_info["likelihood_state"])

    gt_pred = GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)
else:
    gt_pred = None

batch = SyntheticBatch(
    x=x, y=y, xt=xt, yt=yt, xc=xc, yc=yc,
    gt_pred=gt_pred
)

print(batch.x[:10])
print(batch.y[:10])
print(batch.gt_pred.kernel)