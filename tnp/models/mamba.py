from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn
import copy

from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerEncoder
from ..networks.mamba import MNPNDMambaEncoder, TNPMambaEncoder
from .base import CausalNeuralProcess
from tnp.utils.helpers import preprocess_observations, preprocess_contexts, preprocess_targets
from tnp.networks.mamba import assign_mamba_layer_indices, create_inference_params_cache


class FullSequenceDecoder(nn.Module):
    """
    Decodes the entire sequence of representations (z) without slicing.
    Used for causal/temporal models where loss is calculated on all points.
    """
    def __init__(self, z_decoder: nn.Module):
        super().__init__()
        self.z_decoder = z_decoder

    @check_shapes("z: [m, ..., n, dz]", "return: [m, ..., n, dy]")
    def forward(
        self, z: torch.Tensor, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # We ignore xt because we want the model to decode EVERY point 
        # provided by the encoder (both context and target).
        return self.z_decoder(z)



class CausalTemporalMambaEncoder(nn.Module):
    def __init__(
        self, 
        mamba_layer: nn.Module, 
        xy_encoder: nn.Module,  # Add this
        num_layers: int, 
    ):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(mamba_layer) for _ in range(num_layers)])
        self.xy_encoder = xy_encoder # Use this instead of input_projection
        assign_mamba_layer_indices(self)

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        #yc, yt = preprocess_observations(xt, yc)
        full_x = torch.cat([xc, xt], dim=1)  
        full_y_0 = torch.cat([yc, yt], dim=1) 

        full_y, _ = preprocess_observations(xt, full_y_0) 

        b, n, _ = full_y.shape
        zeros = torch.zeros(b, 1, full_y.shape[-1], device=full_y.device)
        shifted_y = torch.cat([zeros, full_y[:, :-1, :]], dim=1) 

        tokens = torch.cat([full_x, shifted_y], dim=-1) 
        
        # Use the MLP xy_encoder here!
        z = self.xy_encoder(tokens) 

        for layer in self.layers:
            z = layer(z)
        return z

    @torch.no_grad()
    @check_shapes("xc: [m, nc, d]", "yc: [m, nc, dy]", "return: [m, nc, dz]")
    def process_context_for_ar(self, xc: torch.Tensor, yc: torch.Tensor, inf_cache) -> (torch.Tensor, torch.Tensor):
        # This method can be used to sample yt autoregressively during inference
        yc = preprocess_contexts(yc)
        zeroes = torch.zeros(xc.shape[0], 1, yc.shape[-1], device=yc.device)
        shifted_yc = torch.cat([zeroes, yc[:, :-1, :]], dim=1) 

        tokens = torch.cat([xc, shifted_yc], dim=-1) 
        z = self.xy_encoder(tokens)

        for layer in self.layers:
            z = layer(z, inference_params=inf_cache)
        
        inf_cache.seqlen_offset += xc.shape[1]
        return z

    @torch.no_grad()
    @check_shapes(
        "next_x: [m, 1, dx]", "prev_y: [m, 1, dy]", "return: [m, 1, dz]"
    )
    def sample_next_ar(self, next_x: torch.Tensor, prev_y: torch.Tensor, inf_cache) -> torch.Tensor:
        assert next_x.shape[1] == 1, "sample_next_ar is designed to sample one target point at a time."
        prev_y = preprocess_contexts(prev_y)
        token = torch.cat([next_x, prev_y], dim=-1) 
        z = self.xy_encoder(token)

        for layer in self.layers:
            z = layer(z, inference_params=inf_cache)
        inf_cache.seqlen_offset += z.shape[1]

        return z[:, -1:, :]


class MAMBA(CausalNeuralProcess):
    def __init__(
        self,
        encoder: CausalTemporalMambaEncoder,
        decoder: FullSequenceDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)

    @torch.no_grad()
    def sample_yt_ar(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        mamba_cache = create_inference_params_cache(max_seqlen = xc.shape[1] + xt.shape[1], max_batch_size = xc.shape[0])
        self.encoder.process_context_for_ar(xc, yc, mamba_cache)
        prev_y = yc[:, -1:, :]  # Start with the last context point's y
        sampled_yt_list = []
        for i in range(xt.shape[1]):
            next_z = self.encoder.sample_next_ar(xt[:, i:i+1, :], prev_y, mamba_cache)
            next_y_dist = self.likelihood(self.decoder(next_z))
            next_y = next_y_dist.sample()
            sampled_yt_list.append(next_y)
            prev_y = next_y
        return torch.cat(sampled_yt_list, dim=1)