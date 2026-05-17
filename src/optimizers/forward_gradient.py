"""
Forward Gradient Descent (FGD)

Источник:
    Baydin, Pearlmutter, Syme, Wood, Torr.
    "Gradients without Backpropagation." arXiv:2202.08587 (2022).
    https://github.com/orobix/fwdgrad
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)


Closure = Callable[[Tuple[torch.Tensor, ...]], torch.Tensor]


class ForwardGradient(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        device = self._all_params()[0].device
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self.last_info: dict[str, object] = {}

    def _all_params(self) -> list[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    @torch.no_grad()
    def step(
        self,
        closure: Closure,
        *,
        projector: BaseProjector | None = None,
        projection: ProjectionMode = "none",
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError(
                "ForwardGradient requires a closure that maps a tuple of "
                "parameters to a scalar loss."
            )
        if projection != "none" and projector is None:
            raise ValueError("projection is not 'none', but projector is None.")

        all_params = self._all_params()
        primals = tuple(p.detach() for p in all_params)
        tangents = tuple(
            torch.randn(
                p.shape, device=p.device, dtype=p.dtype, generator=self._generator
            )
            if p.requires_grad
            else torch.zeros_like(p)
            for p in all_params
        )

        with torch.enable_grad():
            loss, jvp_val = torch.func.jvp(closure, (primals,), (tangents,))
        directional = float(jvp_val.detach())

        raw_update = tuple(
            directional * v if p.requires_grad else torch.zeros_like(p)
            for p, v in zip(all_params, tangents)
        )

        update, info = project_or_passthrough(raw_update, projector, projection)

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "directional": directional,
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals,
        }

        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                u = update[idx]
                idx += 1
                if not p.requires_grad:
                    continue
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(u, alpha=-lr)

        return loss.detach()
