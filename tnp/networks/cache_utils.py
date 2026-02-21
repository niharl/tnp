# Helper to make KV cache updates easy
from typing import Optional
import torch


# Updates context rep stored
def update_ctx_cache(zc_new, cache, cache_id):
    zc_old = cache.get(cache_id, None)

    # Adds new representation
    if zc_old is not None:
        zc_new = torch.cat((zc_old, zc_new), dim=1)
    cache[cache_id] = zc_new