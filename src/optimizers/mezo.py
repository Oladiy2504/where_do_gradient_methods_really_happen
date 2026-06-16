from __future__ import annotations

import warnings
from typing import Callable, Iterable

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)
from src.projections.base import _tree_norm, _unflatten_like


class MeZO(Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-6,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int | None = None,
        subspace_sampling: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        device = self._all_params()[0].device
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self.subspace_sampling = bool(subspace_sampling)
        self.last_info: dict[str, object] = {}

    def _all_params(self) -> list[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _sample_one(self, p: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            p.shape,
            device=p.device,
            dtype=p.dtype,
            generator=self._generator,
        )

    @torch.no_grad()
    def _shift_inplace(self, gen_state: torch.Tensor, alpha: float) -> None:
        self._generator.set_state(gen_state)
        for p in self._all_params():
            if not p.requires_grad:
                continue
            z = self._sample_one(p)
            p.data.add_(z, alpha=alpha)

    @torch.no_grad()
    def _materialize_z(self, gen_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._generator.set_state(gen_state)
        out: list[torch.Tensor] = []
        for p in self._all_params():
            if not p.requires_grad:
                out.append(torch.zeros_like(p))
            else:
                out.append(self._sample_one(p))
        return tuple(out)

    def _materialize_z_from_subspace(
        self, projector: BaseProjector
    ) -> tuple[torch.Tensor, ...]:
        Q = projector.basis
        xi = torch.randn(
            Q.shape[1],
            device=Q.device,
            dtype=Q.dtype,
            generator=self._generator,
        )
        z_flat = Q @ xi

        trainable_params = [p for p in self._all_params() if p.requires_grad]
        z_trainable = _unflatten_like(z_flat, trainable_params)

        out: list[torch.Tensor] = []
        t_idx = 0
        for p in self._all_params():
            if p.requires_grad:
                out.append(z_trainable[t_idx].to(device=p.device, dtype=p.dtype))
                t_idx += 1
            else:
                out.append(torch.zeros_like(p))
        return tuple(out)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        *,
        projector: BaseProjector | None = None,
        projection: ProjectionMode = "none",
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("MeZO requires a closure that returns a scalar loss.")
        if projection != "none" and projector is None:
            raise ValueError("projection is not 'none', but projector is None.")

        eps = self.param_groups[0]["eps"]
        gen_state = self._generator.get_state()

        use_projection = projector is not None and projection != "none"
        use_subspace_sampling = (
            self.subspace_sampling
            and use_projection
            and getattr(projector, "basis", None) is not None
        )

        if self.subspace_sampling and not use_subspace_sampling:
            warnings.warn(
                "MeZO.subspace_sampling=True requires a projector with a ready "
                "basis and projection != 'none'; falling back to full-space "
                "sampling for this step.",
                RuntimeWarning,
                stacklevel=2,
            )

        z: tuple[torch.Tensor, ...] | None
        if use_subspace_sampling:
            z = self._materialize_z_from_subspace(projector)
        elif use_projection:
            z = self._materialize_z(gen_state)
        else:
            z = None

        if z is not None:
            for p, zi in zip(self._all_params(), z):
                if p.requires_grad:
                    p.data.add_(zi, alpha=eps)
        else:
            self._shift_inplace(gen_state, alpha=+eps)

        with torch.enable_grad():
            loss_plus = closure()
        loss_plus_val = float(loss_plus.detach())

        if z is not None:
            for p, zi in zip(self._all_params(), z):
                if p.requires_grad:
                    p.data.add_(zi, alpha=-2.0 * eps)
        else:
            self._shift_inplace(gen_state, alpha=-2.0 * eps)

        with torch.enable_grad():
            loss_minus = closure()
        loss_minus_val = float(loss_minus.detach())

        if z is not None:
            for p, zi in zip(self._all_params(), z):
                if p.requires_grad:
                    p.data.add_(zi, alpha=+eps)
        else:
            self._shift_inplace(gen_state, alpha=+eps)

        coeff = (loss_plus_val - loss_minus_val) / (2.0 * eps)

        if use_subspace_sampling:

            train_z = [zi for p, zi in zip(self._all_params(), z) if p.requires_grad]
            z_norm = float(_tree_norm(train_z).cpu())
            raw_norm = abs(coeff) * z_norm

            self.last_info = {
                "projection": projection,
                "mezo_coeff": coeff,
                "raw_update_norm": raw_norm,
                "projected_update_norm": raw_norm,
                "alignment": 1.0,
                "eigvals": getattr(projector, "eigvals", None)
            }

            idx = 0
            for group in self.param_groups:
                lr = group["lr"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    zi = z[idx]
                    idx += 1
                    if not p.requires_grad:
                        continue
                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(zi, alpha=-lr * coeff)

        elif use_projection:
            raw_update = tuple(coeff * zi for zi in z)
            update, info = project_or_passthrough(raw_update, projector, projection)

            self.last_info = {
                "projection": projection,
                "mezo_coeff": coeff,
                "raw_update_norm": info.raw_norm,
                "projected_update_norm": info.projected_norm,
                "alignment": info.alignment,
                "eigvals": info.eigvals
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
        else:
            self._generator.set_state(gen_state)

            z_sq_sum = 0.0
            for group in self.param_groups:
                lr = group["lr"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    zi = self._sample_one(p)
                    z_sq_sum += float(zi.detach().pow(2).sum().cpu())
                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(zi, alpha=-lr * coeff)

            raw_norm = abs(coeff) * (z_sq_sum ** 0.5)
            self.last_info = {
                "projection": "none",
                "mezo_coeff": coeff,
                "raw_update_norm": raw_norm,
                "projected_update_norm": raw_norm,
                "alignment": 1.0,
                "eigvals": None
            }

        return torch.tensor(
            0.5 * (loss_plus_val + loss_minus_val),
            device=self._all_params()[0].device,
        )
