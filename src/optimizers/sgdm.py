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


class SGDM(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
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
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p))
                    continue

                grad = p.grad.float()
                if wd != 0.0:
                    grad = grad.add(p.float(), alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.clone(grad).detach().float()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                direction = grad.add(buf, alpha=momentum) if nesterov else buf
                raw_update.append(lr * direction)

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
            p.add_(u.to(dtype=p.dtype), alpha=-1.0)

        return loss
