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


class ReversedGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
            self, 
            base_gt_pred: GPGroundTruthPredictor,
            reversal_point: float, 
            context_range: torch.Tensor, 
            **kwargs):
        super().__init__(**kwargs)
        self.base_gt_pred = base_gt_pred
        self.reversal_point = reversal_point
        self.context_range = context_range
        self.min_context = context_range[:, 0]
        self.max_context = context_range[:, 1]
    
    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Ensure boundary tensors are on the same device as inputs
        min_c = self.min_context.to(xc.device)
        max_c = self.max_context.to(xc.device)
        reversal_point = self.reversal_point

        def transform_inputs(x):
            # Create a boolean mask: True where elements are inside the original context range
            # Broadcasting handles shape (Batch, N, Dim) vs (Dim,)
            is_in_range = (x >= min_c) & (x <= max_c)
            
            # Calculate the reflection for ALL points: x' = 2 * r - x
            x_reflected = 2 * reversal_point - x
            
            # Select element-wise and flip inputs outside of context range
            return torch.where(is_in_range, x, x_reflected)
        
        # Apply transformation to both context and target inputs
        xc_transformed = transform_inputs(xc)
        xt_transformed = transform_inputs(xt)

        # Call the parent GPGroundTruthPredictor with the transformed inputs.
        return self.base_gt_pred.__call__(xc_transformed, yc, xt_transformed, yt)

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        pass

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
    ) -> Tuple[torch.Tensor, GPGroundTruthPredictor]:
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
        same_targets: bool = True, # whether target points are same as context points reversed
        shared_noise: bool = True, # whether the noise is also shared between context and target
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_point = reversal_point
        self.same_targets = same_targets
        self.shared_noise = shared_noise

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())

        if self.same_targets:
            nt = nc
        else:
            nt = torch.randint(low=self.min_nt, high=self.max_nt + 1, size=())

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
        
        # Randomly flip the context range around the reversal point
        if torch.rand(1) > 0.5:
            current_range = 2 * self.reversal_point - self.context_range
            current_range = current_range.flip(dims=[1])
        else:
            current_range = self.context_range

        # Sample context inputs
        xc = self.sample_inputs(
            n=nc,
            context_range=current_range,
            batch_shape=batch_shape)

        if self.same_targets and self.shared_noise:
            # Create target inputs by reversing context inputs around reversal_point
            xt = 2 * self.reversal_point - xc
            yc, non_reversed_gt_pred = self.sample_outputs(x=xc)
            yt = yc

        elif not self.shared_noise:
            if self.same_targets:
                xt = 2 * self.reversal_point - xc
                xquery = torch.concat([xc, xc], axis = 1)
            else:
                xt_reversed = self.sample_inputs(
                    n=nt,
                    context_range=current_range,
                    batch_shape=batch_shape)
                xt = 2 * self.reversal_point - xt_reversed
                xquery = torch.concat([xc, xt_reversed], axis = 1)
                
            yquery, non_reversed_gt_pred = self.sample_outputs(x=xquery)
            yc = yquery[:, :nc, :]
            yt = yquery[:, nc:, :]

        else:
            raise NotImplementedError("Noise can only be shared if targets are flipped contexts.")

        x = torch.concat([xc, xt], axis=1)
        y = torch.concat([yc, yt], axis=1)

        reversed_gt_pred = ReversedGPGroundTruthPredictor(
            base_gt_pred=non_reversed_gt_pred,
            reversal_point=self.reversal_point,
            context_range=current_range
        )
        
        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=reversed_gt_pred,
        )
    
    def sample_inputs(
        self,
        n: int,
        context_range: torch.Tensor,
        batch_shape: torch.Size,
    ) -> torch.Tensor:

        # Sample context inputs
        xc = (
            torch.rand((*batch_shape, n, self.dim))
            * (context_range[:, 1] - context_range[:, 0])
            + context_range[:, 0]
        )

        return xc
    
class RandomReversalGPGenerator(RandomScaleGPGenerator):
    """
    Generates batches of GP data with randomised reversal point, and 
    randomised size of the context range
    """

    def __init__(
        self,
        *,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        batch_size: int,
        reversal_range: Tuple[float, float],
        priming_frac_range: Tuple[float, float],
        same_targets: bool = True,
        shared_noise: bool = True, 
        **kwargs,
    ):
        super().__init__(min_nc=min_nc, max_nc=max_nc, min_nt = min_nt, 
                         max_nt = max_nt, batch_size=batch_size, **kwargs)
        self.reversal_range = reversal_range
        self.priming_frac_range = priming_frac_range
        self.same_targets = same_targets
        self.shared_noise = shared_noise

    def generate_batch(self) -> SyntheticBatch:
        # Sample number of context = number of target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = nc

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
        # Sample random reversal point within specified range
        self.reversal_point = float(torch.rand(1)) * (self.reversal_range[1] - self.reversal_range[0]) + self.reversal_range[0]

        # for now, the context range is clipped at the reversal point
        current_range = self.context_range.clone()
        current_range[:, 1] = self.reversal_point

        # Sample non_flipped inputs
        x_base = self.sample_inputs(
            nc=nc,
            context_range=current_range,
            batch_shape=batch_shape)
        x_base, _ = torch.sort(x_base, dim = -2)
        y_base, non_reversed_gt_pred = self.sample_outputs(x=x_base)

        # Create target inputs by reversing context inputs around reversal_point
        x_flipped = 2 * self.reversal_point - x_base
        x_flipped = x_flipped.flip(dims=[-2])
        y_flipped = y_base.flip(dims=[-2])

        # Create full inputs and outputs by concatenation
        x = torch.concat([x_base, x_flipped], axis=1)
        y = torch.concat([y_base, y_flipped], axis=1)

        # Split into context and target sets
        priming_frac_low = self.priming_frac_range[0]
        priming_frac_high = self.priming_frac_range[1]
        priming_frac = float(torch.rand(1)) * (priming_frac_high - priming_frac_low) + priming_frac_low
        n_priming = int(priming_frac * nc)
        
        if self.reversal_point >= 0.0:
            xc = x[:, :nc+n_priming, :]
            yc = y[:, :nc+n_priming, :]
            xt = x[:, nc+n_priming:, :]
            yt = y[:, nc+n_priming:, :]

        else:
            xc = x[:, nc-n_priming:, :]
            yc = y[:, nc-n_priming:, :]
            xt = x[:, :nc-n_priming, :]
            yt = y[:, :nc-n_priming, :]


        # Create the reversed GP predictor
        reversed_gt_pred = ReversedGPGroundTruthPredictor(
            base_gt_pred=non_reversed_gt_pred,
            reversal_point=self.reversal_point,
            context_range=current_range
        )

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=reversed_gt_pred,
        )
    
    def sample_inputs(
        self,
        nc: int,
        context_range: torch.Tensor,
        batch_shape: torch.Size,
    ) -> torch.Tensor:

        # Sample context inputs
        xc = (
            torch.rand((*batch_shape, nc, self.dim))
            * (context_range[:, 1] - context_range[:, 0])
            + context_range[:, 0]
        )

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
