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



class TNPMambaEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mamba_layer: MambaEncoderLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mamba_layers = _get_clones(mamba_layer, num_layers)
    
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
            xt = mhca_layer(xt, xc)

        return xt

class MNP_NDMambaEncoder(nn.Module):
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
        self, xc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:

        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)

        return xt