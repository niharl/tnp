import torch
from torch import nn

from ..data.base import Batch, ImageBatch
from ..models.base import (
    ARConditionalNeuralProcess,
    ConditionalNeuralProcess,
    LatentNeuralProcess,
    CausalNeuralProcess,
)
from ..models.convcnp import GriddedConvCNP


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.distributions.Distribution:
    if isinstance(model, GriddedConvCNP):
        assert isinstance(batch, ImageBatch)
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, LatentNeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    elif isinstance(model, ARConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif isinstance(model, CausalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    else:
        raise ValueError

    return pred_dist


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    pred_dist = np_pred_fn(model, batch, num_samples)
    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik


def full_sequence_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Calculates NLL over the entire sequence (context + target).
    """
    # 1. Get predictions (Nc + Nt points if using FullSequenceDecoder)
    pred_dist = np_pred_fn(model, batch)
    
    # 2. Concatenate context and target ground truths
    y_all = torch.cat([batch.yc, batch.yt], dim=1)
    
    # 3. Calculate log-likelihood against the full sequence
    loglik = pred_dist.log_prob(y_all).sum() / y_all[..., 0].numel()

    return -loglik


def ar_loss_fn(
    model: nn.Module,
    batch: Batch,
):
    """
    Calculates NLL over the target sequence only.
    """

    if isinstance(model, CausalNeuralProcess):
        # 1. Get predictions (Nc + Nt points if using FullSequenceDecoder)
        pred_dist = np_pred_fn(model, batch)
        
        # 2. Concatenate context and target ground truths
        y_all = torch.cat([batch.yc, batch.yt], dim=1)
        
        # 3. Calculate log-probabilities for the entire sequence
        log_probs = pred_dist.log_prob(y_all)
        
        # 4. Slice to isolate only the target values (last Nt points)
        nt = batch.yt.shape[1]
        target_log_probs = log_probs[:, -nt:]
        
        # 5. Calculate negative log-likelihood against the target sequence
        loglik = target_log_probs.sum() / batch.yt[..., 0].numel()
    
    elif isinstance(model, ConditionalNeuralProcess):
        # 1. Initialise the incremental caching structures
        model.init_inc_structs()
        
        # 2. Prime the cache with the entire context sequence
        model.update_ctx(batch.xc, batch.yc)
        
        nt = batch.xt.shape[1]
        total_loglik = 0.0
        
        # 3. Iteratively query and update for each target point
        for i in range(nt):
            # Isolate the i-th target point
            xt_i = batch.xt[:, i:i+1, :]
            yt_i = batch.yt[:, i:i+1, :]
            
            # Query the model for the predictive distribution of the current target
            pred_dist_i = model.query(xt_i)
            
            # Accumulate the log-likelihood for this point
            total_loglik += pred_dist_i.log_prob(yt_i).sum()
            
            # Update the context cache with the newly "observed" target point
            model.update_ctx(xt_i, yt_i)
            
        # 4. Average the log-likelihood over all target points
        loglik = total_loglik / batch.yt[..., 0].numel()

    else:
        raise ValueError("ar_loss_fn is only implemented for CausalNeuralProcess and ConditionalNeuralProcess.")

    return -loglik


def ar_pred_fn(
    model: nn.Module,
    batch: Batch,
) -> torch.Tensor:
    """
    Calculates NLL over the target sequence only.
    """

    if isinstance(model, CausalNeuralProcess):
        pred_dist = np_pred_fn(model, batch)
        nt = batch.yt.shape[1]
        # Extract and slice the underlying parameters
        sliced_loc = pred_dist.loc[:, -nt:, :]
        sliced_scale = pred_dist.scale[:, -nt:, :]
        # Return a new Normal distribution using the sliced parameters
        return torch.distributions.Normal(sliced_loc, sliced_scale)
    
    elif isinstance(model, ConditionalNeuralProcess):
        # 1. Initialise the incremental caching structures
        model.init_inc_structs()
        
        # 2. Prime the cache with the entire context sequence
        model.update_ctx(batch.xc, batch.yc)
        
        nt = batch.xt.shape[1]
        total_loglik = 0.0

        # Store parameters instead of distribution objects
        locs = []
        scales = []
        
        # 3. Iteratively query and update for each target point
        for i in range(nt):
            # Isolate the i-th target point
            xt_i = batch.xt[:, i:i+1, :]
            yt_i = batch.yt[:, i:i+1, :]
            
            # Query the model for the predictive distribution of the current target
            pred_dist_i = model.query(xt_i)
            
            # Extract mean (loc) and standard deviation (scale)
            locs.append(pred_dist_i.loc)
            scales.append(pred_dist_i.scale)
            
            # Update the context cache with the newly "observed" target point
            model.update_ctx(xt_i, yt_i)
            
        # Stack the parameters along the time sequence dimension (dim=1)
        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)
        
        # Return a new batched Normal distribution
        return torch.distributions.Normal(locs, scales)

    else:
        raise ValueError("ar_pred_fn is only implemented for CausalNeuralProcess and ConditionalNeuralProcess.")


def sample_function_trajectory(
    model: nn.Module,
    batch: Batch,
) -> torch.Tensor:
    """
    Samples a trajectory from the model, i.e. samples yt autoregressively.
    """
    if isinstance(model, CausalNeuralProcess):
        return model.sample_yt_ar(batch.xc, batch.yc, batch.xt)

    elif isinstance(model, ConditionalNeuralProcess):
        model.init_inc_structs()
        model.update_ctx(batch.xc, batch.yc)
        nt = batch.xt.shape[1]
        sampled_yt_list = []
        for i in range(nt):
            xt_i = batch.xt[:, i:i+1, :]
            pred_dist_i = model.query(xt_i)
            sampled_yt_i = pred_dist_i.sample()
            sampled_yt_list.append(sampled_yt_i)
            model.update_ctx(xt_i, sampled_yt_i)
        return torch.cat(sampled_yt_list, dim=1)

    else:
        raise ValueError("sample_function_trajectory is only implemented for CausalNeuralProcess and ConditionalNeuralProcess.")