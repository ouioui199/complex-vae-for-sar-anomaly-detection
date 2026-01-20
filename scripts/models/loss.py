from types import NoneType

import torch


def complex_kullback_leibler_divergence_loss(mu: torch.Tensor, sigma: torch.Tensor, delta: torch.Tensor, weight: float | NoneType = None) -> torch.Tensor:
    loss = torch.mean(
        torch.sum(
            torch.abs(mu) ** 2 + sigma - 1 - 0.5 * torch.log(sigma ** 2 - torch.abs(delta) ** 2),
        dim = (1, 2, 3)),
    dim = 0)
    
    if weight is not None:
        return weight * loss
    return loss
