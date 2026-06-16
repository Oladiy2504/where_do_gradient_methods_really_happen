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


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        self.last_info: dict[str, object] = {}

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Closure] = None,
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
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p))
                    continue
                d_p = p.grad
                if wd != 0.0:
                    d_p = d_p.add(p, alpha=wd)
                raw_update.append(lr * d_p)

        update, info = project_or_passthrough(tuple(raw_update), projector, projection)

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals
        }

        for p, u in zip(all_params, update):
            if p.grad is None:
                continue
            p.add_(u, alpha=-1.0)

        return loss
