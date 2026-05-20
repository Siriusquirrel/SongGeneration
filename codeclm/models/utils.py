# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch

def sample_top_k(probs: torch.Tensor, k: int, generator=None) -> torch.Tensor:
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
    k_offset = multinomial(top_k_probs, 1, generator=generator)
    return torch.gather(top_k_indices, -1, k_offset)

def sample_top_p(probs: torch.Tensor, p: float, generator=None) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    sorted_idx = multinomial(probs_sort, num_samples=1, generator=generator)
    return torch.gather(probs_idx, -1, sorted_idx)

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    input_flat = input.reshape(-1, input.shape[-1])
    samples_flat = torch.multinomial(input_flat, num_samples=num_samples, replacement=replacement, generator=generator)
    return samples_flat.reshape(*input.shape[:-1], num_samples)
