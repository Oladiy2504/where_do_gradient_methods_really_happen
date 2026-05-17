from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)


Closure = Callable[[], torch.Tensor]


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_info: dict[str, object] = {}

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        projector: BaseProjector | None = None,
        projection: ProjectionMode = "none",
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if projection != "none" and projector is None:
            raise ValueError("projection is not 'none', but projector is None.")

        all_params = [p for group in self.param_groups for p in group["params"]]
        raw_update: list[torch.Tensor] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p))
                    continue

                # Decoupled weight decay sits outside the projection.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                grad = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step_size = lr * ((1.0 - beta2 ** t) ** 0.5) / (1.0 - beta1 ** t)
                denom = exp_avg_sq.sqrt().add_(eps)
                raw_update.append(step_size * exp_avg / denom)

        update, info = project_or_passthrough(tuple(raw_update), projector, projection)

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals,
        }

        for p, u in zip(all_params, update):
            if p.grad is None:
                continue
            p.add_(u.to(dtype=p.dtype), alpha=-1.0)

        return loss
