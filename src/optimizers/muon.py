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


def _muon_orthogonalize(
    x: torch.Tensor, steps: int = 5, eps: float = 1e-7, full_ort: bool = False
) -> torch.Tensor:
    """
    Ортогонализация Ньютона–Шульца для тензоров
    """
    if x.ndim < 2:
        return x

    original_shape = x.shape
    original_dtype = x.dtype

    if x.ndim > 2:
        m = x.reshape(x.shape[0], -1)
    else:
        m = x

    y = m.float()

    if full_ort:
        u, _, vh = torch.linalg.svd(y, full_matrices=False)
        y = u @ vh
        return y.to(dtype=original_dtype).reshape(original_shape)

    a, b, c = 3.4445, -4.7750, 2.0315

    y = y / (y.norm() + eps)

    transposed = False
    if y.shape[0] > y.shape[1]:
        y = y.t()
        transposed = True

    for _ in range(steps):
        yy = y @ y.t()
        y = a * y + (b * yy + c * (yy @ yy)) @ y

    if transposed:
        y = y.t()

    return y.to(dtype=original_dtype).reshape(original_shape)


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        full_ort: bool = False,
        orth_after_projection: bool = True,
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
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            full_ort=full_ort,
            orth_after_projection=orth_after_projection,
        )
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
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p, dtype=torch.float32))
                    continue

                grad = p.grad.float()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                direction = grad.add(buf, alpha=momentum) if nesterov else buf

                raw_update.append(direction)

        per_param_settings: list[tuple[int, bool, bool]] = []
        for group in self.param_groups:
            for _ in group["params"]:
                per_param_settings.append(
                    (
                        group["ns_steps"],
                        group["full_ort"],
                        group["orth_after_projection"],
                    )
                )

        if any(not s[2] for s in per_param_settings):
            pre_proj: list[torch.Tensor] = []
            for p, u, (ns_steps, full_ort, _) in zip(
                all_params, raw_update, per_param_settings
            ):
                if p.grad is not None and p.ndim >= 2:
                    pre_proj.append(
                        _muon_orthogonalize(u, steps=ns_steps, full_ort=full_ort)
                    )
                else:
                    pre_proj.append(u)
            update, info = project_or_passthrough(
                tuple(pre_proj), projector, projection
            )
        else:
            update, info = project_or_passthrough(
                tuple(raw_update), projector, projection
            )

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals,
        }

        idx = 0

        for group in self.param_groups:
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            full_ort = group["full_ort"]
            orth_after_projection = group["orth_after_projection"]

            for p in group["params"]:
                u = update[idx]
                idx += 1

                if p.grad is None:
                    continue

                if orth_after_projection and p.ndim >= 2:
                    u = _muon_orthogonalize(u, steps=ns_steps, full_ort=full_ort)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(u.to(dtype=p.dtype), alpha=-lr)

        return loss
