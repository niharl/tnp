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

@check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
def sort_context_target_separately(
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sorts context (xc, yc) and target (xt) points independently based on x values.
    """
    # Get indices that sort based on the first feature of xc
    c_sort_idxs = torch.argsort(xc[..., 0], dim=-1)

    # Expand indices to match dimensions of xc and yc
    # [m, nc] -> [m, nc, dx]
    c_idxs_x = c_sort_idxs.unsqueeze(-1).expand(-1, -1, xc.shape[-1])
    # [m, nc] -> [m, nc, dy]
    c_idxs_y = c_sort_idxs.unsqueeze(-1).expand(-1, -1, yc.shape[-1])

    sorted_xc = torch.gather(xc, 1, c_idxs_x)
    sorted_yc = torch.gather(yc, 1, c_idxs_y)

    # Get indices that sort based on the first feature of xt
    t_sort_idxs = torch.argsort(xt[..., 0], dim=-1)

    # Expand indices to match dimensions of xt
    t_idxs_x = t_sort_idxs.unsqueeze(-1).expand(-1, -1, xt.shape[-1])
    
    sorted_xt = torch.gather(xt, 1, t_idxs_x)

    return sorted_xc, sorted_yc, sorted_xt