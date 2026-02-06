import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import (
    MultiHeadCrossAttentionLayer
)
from .mamba_layers import MambaEncoderLayer
from .transformer import _get_clones

class InferenceParams:
    """Minimal container for Mamba states."""
    def __init__(self, max_seqlen, batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = batch_size
        self.seqlen_offset = 0
        self.key_value_memory_dict = {}

class MNPDMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mamba_layer: MambaEncoderLayer,
    ):
        super().__init__()

        self.mamba_layers = _get_clones(mamba_layer, num_layers)
    
    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = xc.shape[0]
        inference_params = InferenceParams(max_seqlen=2048 # this is irrelevant/unused
                                           , batch_size=batch_size)
        inference_params.seqlen_offset = 0

        layer_idx = 0
        for mamba_layer in self.mamba_layers:
            if mask is not None:
                warnings.warn("mask is not currently being used.")

            mamba_layer.set_layer_idx(layer_idx)
            # pass contexts through Mamba encoder layers
            xc = mamba_layer(xc, inference_params=inference_params)
            
            # step outputs of Mamba encoder layers for each target point
            xt = mamba_layer.step_independent(xt, inference_params)

            layer_idx += 1
            
        return xt

class MNPNDMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mamba_layer: MambaEncoderLayer,
    ):
        super().__init__()

        self.mamba_layers = _get_clones(mamba_layer, num_layers)
    
    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x = torch.concat((xc, xt), dim=1)
        for mamba_layer in self.mamba_layers:
            if mask is not None:
                warnings.warn("mask is not currently being used.")
            x = mamba_layer(x)

        xt = x[:, xc.shape[1]:, :]
        return xt

class TNPMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mamba_layer: MambaEncoderLayer,
        cross_attention_dilation = False, # dilation of input embeddings for cross-attention
        dilation_factor = 1,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mamba_layers = _get_clones(mamba_layer, num_layers)
        self.ca_dilation = cross_attention_dilation
        self.dilation_factor = dilation_factor
    
    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        for mamba_layer, mhca_layer in zip(self.mamba_layers, self.mhca_layers):
            if mask is not None:
                warnings.warn("mask is not currently being used.")

            xc = mamba_layer(xc)

            if self.ca_dilation:
                nc = xc.shape[1]
                start_index = (nc - 1) % self.dilation_factor
                xc_dilated = xc[:, start_index::self.dilation_factor, :]
                xt = mhca_layer(xt, xc_dilated)
            else:
                xt = mhca_layer(xt, xc)

        return xt