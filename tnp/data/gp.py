import random
from abc import ABC
from typing import Dict, Iterable, Optional, Tuple, Union

import einops
import gpytorch
import torch

from ..networks.gp import RandomHyperparameterKernel
from .base import GroundTruthPredictor
from .synthetic import SyntheticGeneratorUniformInput, SyntheticBatch


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        train_inputs: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        self.kernel = kernel
        self.likelihood = likelihood

        self._result_cache: Optional[Dict[str, torch.Tensor]] = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Move devices.
        old_device = xc.device
        device = self.kernel.device
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad():

                dist = gp_model(xt_)
                pred_dist = gp_model.likelihood.marginal(dist)
                if yt is not None:
                    gt_loglik = pred_dist.to_data_independent_dist().log_prob(
                        yt[i, ..., 0]
                    )
                    gt_loglik_list.append(gt_loglik)

                mean_list.append(pred_dist.mean)
                try:
                    std_list.append(pred_dist.stddev)
                except RuntimeError:
                    std_list.append(pred_dist.covariance_matrix.diagonal() ** 0.5)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        # Cache for deterministic validation batches.
        # Note yt is not specified when passing x_plot.
        if yt is not None:
            self._result_cache = {
                "mean": mean,
                "std": std,
                "gt_loglik": gt_loglik,
            }

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xt = xt.to(old_device)
        if yt is not None:
            yt = yt.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:

        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=self.kernel,
        )
        gp_model.eval()
        gp_model.likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = gp_model.likelihood(f)
            y = dist.sample()
            return y[..., None]


class GPGenerator(ABC):
    def __init__(
        self,
        *,
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = noise_std

    def set_up_gp(self) -> GPGroundTruthPredictor:
        """Pick a random kernel and set up GP predictor."""
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std**2.0

        return GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)

    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        gt_pred = self.set_up_gp()
        y = gt_pred.sample_outputs(x)
        return y, gt_pred


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass

class DeterministicContextGPGenerator(RandomScaleGPGenerator):
    """
    Generates batches of GP data while deterministically increasing
    the number of context points (nc) by a fixed step.
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        nc_step: int,
        min_nt: int,
        max_nt: int,
        nt_step: int,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.current_nc = min_nc
        self.nc_step = nc_step
        self.current_nt = min_nt
        self.nt_step = nt_step

    def generate_batch(self) -> SyntheticBatch:
        """
        Generates a batch with the current number of context points (nc)
        and deterministically increments nc for the next batch.
        """
        nc = self.current_nc
        nt = self.current_nt

        # Sample batch using parent method
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size([self.batch_size])
        )

        return batch

    def increment_lengths(self):
        if self.current_nc + self.nc_step <= self.max_nc:
            self.current_nc += self.nc_step
        if self.current_nt + self.nt_step <= self.max_nt:
            self.current_nt += self.nt_step

    def reset(self):
        """Reset the context points counter to the minimum."""
        self.current_nc = self.min_nc
        self.current_nt = self.min_nt

class ReversedContextGPGenerator(RandomScaleGPGenerator):
    """
    Generates batches of GP data where the GP is reversed at a reversal_point
    and the context points have direct counterparts in the target points.
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_point: float = 0.0,
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_point = reversal_point

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = nc

        # Sample batch using parent method
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size([self.batch_size])
        )

        return batch
    

    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        # Sample context inputs
        xc = self.sample_inputs(nc=nc, batch_shape=batch_shape)
        yc, gt_pred = self.sample_outputs(x=xc)

        # Create target inputs by reversing context inputs around reversal_point
        xt = 2 * self.reversal_point - xc
        yt = yc.flip(dims=[-2])

        x = torch.concat([xc, xt], axis=1)
        y = torch.concat([yc, yt], axis=1)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred
        )
    
    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
    ) -> torch.Tensor:

        # Sample context inputs
        xc = (
            torch.rand((*batch_shape, nc, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            + self.context_range[:, 0]
        )

        # Concatenate context and target inputs
        return xc


class RandomScaleGPGeneratorSameInputs(RandomScaleGPGenerator):

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        x = super().sample_inputs(nc=nc, batch_shape=torch.Size(), nt=nt)
        # copy the inputs b times
        x = einops.repeat(x, "n d -> b n d", b=batch_shape[0])
        return x

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gt_pred = self.set_up_gp()
        sample_shape = x.shape[:-2]
        return gt_pred.sample_outputs(x[0], sample_shape=sample_shape), gt_pred
