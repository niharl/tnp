from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn
import torch.nn.functional as F


class BaseMultiHeadAttention(nn.Module, ABC):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        linear: bool = False,
    ):
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        self.linear = linear

    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.to_q(xq)
        k = self.to_k(xk)
        v = self.to_v(xv)

        q, k, v = map(
            lambda x: einops.rearrange(x, "m n (h d) -> m h n d", h=self.num_heads),
            (q, k, v),
        )

        if mask is not None:
            mask = einops.repeat(mask, "m n1 n2 -> m h n1 n2", h=self.num_heads)

        if self.linear:
            out = linear_attention(q, k, v, attn_mask=mask, scale=self.scale)
        else:
            out = nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                q, k, v, attn_mask=mask, scale=self.scale
            )

        out = einops.rearrange(out, "m h n d -> m n (h d)")
        out = self.to_out(out)
        return out


class MultiHeadAttention(BaseMultiHeadAttention):
    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        return super().propagate(xq, xk, xv, mask)


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().propagate(x, x, x, mask)


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dx]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        return super().propagate(xq, xkv, xkv, mask)


class MultiHeadKRAttention(BaseMultiHeadAttention):
    """https://arxiv.org/abs/2411.12502."""
    def __init__(self, *, embed_dim: int, **kwargs):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nkv, dx]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        # Concatenate queries and keys.
        xqk = torch.cat([xq, xkv], dim=-2)

        out = super().propagate(xqk, xkv, xkv, mask)

        # Split into query and key output.
        outq, outk = torch.split(out, [xq.shape[-2], xkv.shape[-2]], dim=-2)

        return outq, outk


@check_shapes(
    "q: [m, h, nq, dqk]",
    "k: [m, h, nkv, dqk]",
    "v: [m, h, nkv, dq]",
)
def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scale: float = 1.0,
):
    if attn_mask is not None:
        # TODO: What is going on here.
        raise NotImplementedError("Not implemented yet.")

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    q = q * scale

    kv = k.transpose(-1, -2) @ v
    out = q @ kv
    return out

@check_shapes(
    "q: [m, h, nq, dqk]",
    "k: [m, h, nkv, dqk]",
    "v: [m, h, nkv, dv]",
)
def linear_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    eps: float = 1e-6,
):
    """
    Computes Linear Attention using the Katharopoulos method (ELU+1).
    Args:
        q: [m, h, nq, d]
        k: [m, h, nkv, d]
        v: [m, h, nkv, dv]
        attn_mask: Not supported currently.
    """
    # 1. Feature Map (Ensure positivity)
    Q = F.elu(q) + 1.0
    K = F.elu(k) + 1.0
    
    # Apply scale to Q (mimics standard attention scaling)
    Q = Q * scale

    # 2. Let's ignore masking for now
    if attn_mask is not None:
        raise NotImplementedError("Masking not implemented for linear attention yet.")

    # 3. Compute the Global Context Matrix (The "Cache")
    # shape: [m, h, d, dv]
    # This is the O(Nc) step.
    KV = torch.matmul(K.transpose(-1, -2), v)
    
    # 4. Compute the Normaliser (The Denominator)
    # shape: [m, h, d]
    # We sum K over the sequence length.
    Z = K.sum(dim=-2) 
    
    # 5. Compute Output (The O(Nt) step)
    # Numerator: Q @ KV -> [m, h, nq, dv]
    numerator = torch.matmul(Q, KV)
    
    # Denominator: Dot product of Q and Z
    # Explicit element-wise mul + sum
    # [m, h, nq, d] * [m, h, 1, d] -> sum(-1) -> [m, h, nq, 1]
    denominator = torch.sum(Q * Z.unsqueeze(-2), dim=-1, keepdim=True)

    return numerator / (denominator + eps)
