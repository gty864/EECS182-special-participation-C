from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import wandb

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.synthetic import generate_data
from src.models.logistic import LogisticRegression
from src.optim.sgd import sgd_step

logger = get_logger(__name__)

class SGDTrainer:
    def __init__(self, cfg: DictConfig):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._build()

    def _build(self):
        # data
        self.X, self.y = generate_data(
            n_samples=self.cfg.data.n_samples,
            mean=self.cfg.data.mean,
            cov=self.cfg.data.cov,
            seed=self.cfg.seed,
        )
        self.X, self.y = self.X.to(self.device), self.y.to(self.device)

        # model
        self.model = LogisticRegression(
            input_dim=self.cfg.model.input_dim,
            bias=self.cfg.model.bias,
        ).to(self.device)

        # optim state
        self.velocity: Dict[int, torch.Tensor] = {}

        # logging
        wandb.init(
            project=self.cfg.logging.wandb.project,
            entity=self.cfg.logging.wandb.entity,
            config=cfg,
            mode="disabled" if cfg.logging.wandb.project is None else "online",
        )

    def loss_fn(self, logits: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, self.y.float())

    def train(self) -> tuple[list[float], list[float]]:
        losses_gd, losses_mom = [], []
        lr = self.cfg.optim.lr
        mom = self.cfg.optim.momentum
        max_iter = self.cfg.optim.max_iters

        # ---- GD (momentum = 0) ----
        params = list(self.model.parameters())
        for it in range(max_iter):
            self.model.zero_grad()
            logits = self.model(self.X)
            loss = self.loss_fn(logits)
            loss.backward()
            grads = [p.grad for p in params]
            self.velocity = sgd_step(params, grads, lr, momentum=0.0, velocity=self.velocity)
            losses_gd.append(loss.item())

            if (it + 1) % self.cfg.logging.log_every == 0:
                logger.info(f"[GD] iter {it+1:04d} loss {loss.item():.6f}")

        # ---- Momentum ----
        self.model = LogisticRegression(
            input_dim=self.cfg.model.input_dim,
            bias=self.cfg.model.bias,
        ).to(self.device)
        self.velocity = {}
        for it in range(max_iter):
            self.model.zero_grad()
            logits = self.model(self.X)
            loss = self.loss_fn(logits)
            loss.backward()
            grads = [p.grad for p in params]
            self.velocity = sgd_step(params, grads, lr, momentum=mom, velocity=self.velocity)
            losses_mom.append(loss.item())

            if (it + 1) % self.cfg.logging.log_every == 0:
                logger.info(f"[Mom] iter {it+1:04d} loss {loss.item():.6f}")

        wandb.log({"gd_loss": losses_gd[-1], "mom_loss": losses_mom[-1]})
        wandb.finish()
        return losses_gd, losses_mom