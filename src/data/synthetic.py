from typing import Tuple
import torch

def generate_data(
    n_samples: int,
    mean: list,
    cov: list,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian 2-D data with binary label y = (x[:,1] > 0)."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    X = torch.normal(
        mean=torch.tensor(mean),
        std=torch.tensor(cov).diag().sqrt(),
        size=(n_samples, len(mean)),
        generator=rng,
    )
    y = (X[:, 1] > 0).long()
    return X, y