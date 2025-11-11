from typing import Iterable
import torch

def sgd_step(
    params: Iterable[torch.Tensor],
    grads: Iterable[torch.Tensor],
    lr: float,
    momentum: float = 0.0,
    velocity: dict | None = None,
) -> dict:
    """One SGD (with optional momentum) step."""
    if velocity is None:
        velocity = {id(p): torch.zeros_like(p) for p in params}

    for p, g in zip(params, grads):
        v = velocity[id(p)]
        v.mul_(momentum).add_(g, alpha=lr)
        p.sub_(v)
        velocity[id(p)] = v
    return velocity