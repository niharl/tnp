from typing import Tuple

import torch
from check_shapes import check_shapes


@check_shapes("xt: [m, nt, dx]", "yc: [m, nc, dy]")
def preprocess_observations(
    xt: torch.Tensor,
    yc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    yt = torch.zeros(xt.shape[:-1] + yc.shape[-1:]).to(yc)
    yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
    yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)

    return yc, yt

@check_shapes("yc: [m, nc, dy]")
def preprocess_contexts(
    yc: torch.Tensor
 ) -> torch.Tensor:
    yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
    return yc

@check_shapes("xt: [m, nt, dx]", "return: [m, nt, dy]")
def preprocess_targets(
    xt: torch.Tensor,
    dim_y: Tuple[int]
) -> torch.Tensor:
    yt = torch.zeros(xt.shape[:-1] + dim_y).to(xt)
    yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)
    return yt