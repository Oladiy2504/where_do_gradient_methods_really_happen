from __future__ import annotations

import math
from collections import deque
from typing import Callable, Iterable

import torch

from src.projections.base import LowRankBasisProjector, _flatten


Closure = Callable[[], torch.Tensor]


def _grads_from_closure(
        params: Iterable[torch.nn.Parameter],
        loss_closure: Closure,
) -> tuple[torch.Tensor | None, ...]:
    """Autograd gradients of ``loss_closure()`` w.r.t. ``params`` (may contain None)."""
    with torch.enable_grad():
        loss = loss_closure()
        if loss.ndim != 0:
            loss = loss.mean()

        return torch.autograd.grad(
            loss,
            tuple(params),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )


def _grad_or_zero(
        param: torch.Tensor,
        grad: torch.Tensor | None,
        weight_decay: float,
) -> torch.Tensor:
    """Detached float32 gradient (or zeros if None), with optional weight decay."""
    if grad is None:
        grad = torch.zeros_like(param)
    grad = grad.detach().to(device=param.device, dtype=torch.float32)
    if weight_decay != 0.0:
        grad = grad.add(param.detach().to(dtype=torch.float32), alpha=weight_decay)
    return grad


class _AdaptiveLRProjector(LowRankBasisProjector):
    """Base class for projectors built from recent Adam-style vectors."""

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            history_size: int | None = None,
            center: bool = True,
            seed: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)

        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        self.eps = eps
        self.weight_decay = weight_decay
        self.history_size = max(self.k + 1, 2 * self.k) if history_size is None else int(history_size)
        self.center = center

        if self.history_size <= 0:
            raise ValueError(f"history_size must be positive, got {history_size}.")

        self._history: deque[torch.Tensor] = deque(maxlen=self.history_size)
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    def _set_basis_from_vector(self, flat_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._history.append(flat_vec.detach().to(device=self.device, dtype=self.dtype))
        self._set_basis_from_history()

        if self.basis is None or self.eigvals is None:
            raise RuntimeError("Adaptive LR projector failed to build a basis.")

        return self.eigvals, self.basis

    def _compute_grads(self, loss_closure: Closure) -> tuple[torch.Tensor | None, ...]:
        with torch.enable_grad():
            loss = loss_closure()
            if loss.ndim != 0:
                loss = loss.mean()

            return torch.autograd.grad(
                loss,
                self.params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

    def _grad_or_zero(
            self,
            param: torch.Tensor,
            grad: torch.Tensor | None,
    ) -> torch.Tensor:
        if grad is None:
            grad = torch.zeros_like(param)
        grad = grad.detach().to(device=param.device, dtype=torch.float32)
        if self.weight_decay != 0.0:
            grad = grad.add(param.detach().to(dtype=torch.float32), alpha=self.weight_decay)
        return grad

    def _set_basis_from_history(self) -> None:
        X = torch.stack(tuple(self._history), dim=0).to(
            device=self.device,
            dtype=self.dtype,
        )

        if self.center:
            X = X - X.mean(dim=0, keepdim=True)

        row_norms = torch.linalg.vector_norm(X, dim=1)
        X = X[row_norms > self.eps]

        if X.shape[0] == 0:
            basis = self._random_basis()
            eigvals = torch.zeros(self.k, dtype=torch.float32)
            self.set_basis(basis, eigvals=eigvals, orthonormalize=False)
            return

        gram = X @ X.T
        gram = 0.5 * (gram + gram.T)

        vals_cpu, coeffs_cpu = torch.linalg.eigh(
            gram.detach().to(device="cpu", dtype=torch.float32)
        )
        order = torch.argsort(vals_cpu, descending=True)
        vals_cpu = vals_cpu[order]
        coeffs_cpu = coeffs_cpu[:, order]

        rank = min(self.k, int((vals_cpu > self.eps).sum().item()))
        if rank == 0:
            basis = self._random_basis()
            eigvals = torch.zeros(self.k, dtype=torch.float32)
            self.set_basis(basis, eigvals=eigvals, orthonormalize=False)
            return

        vals = vals_cpu[:rank].to(device=self.device, dtype=self.dtype)
        coeffs = coeffs_cpu[:, :rank].to(device=self.device, dtype=self.dtype)
        basis = X.T @ (coeffs / vals.sqrt().clamp_min(self.eps)[None, :])
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        basis = self._complete_basis(basis[:, :rank])

        scale = X.shape[0] - 1 if self.center and X.shape[0] > 1 else X.shape[0]
        eigvals = vals_cpu[:rank] / float(scale)
        eigvals = torch.cat(
            [eigvals, torch.zeros(self.k - rank, dtype=torch.float32)],
            dim=0,
        )

        self.set_basis(basis, eigvals=eigvals, orthonormalize=False)

    def _random_basis(self) -> torch.Tensor:
        basis = torch.randn(
            self.n_params,
            self.k,
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )
        return torch.linalg.qr(basis, mode="reduced")[0]

    def _complete_basis(self, basis: torch.Tensor) -> torch.Tensor:
        if basis.shape[1] == self.k:
            return basis

        extra = torch.randn(
            self.n_params,
            self.k - basis.shape[1],
            device=self.device,
            dtype=self.dtype,
            generator=self._generator,
        )
        extra = extra - basis @ (basis.T @ extra)
        extra = torch.linalg.qr(extra, mode="reduced")[0]
        return torch.cat([basis, extra], dim=1)


class AdaptiveLRSecondMomentProjector(_AdaptiveLRProjector):
    """Projector onto covariance directions of 1 / (sqrt(v) + eps)."""

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            beta2: float = 0.999,
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            history_size: int | None = None,
            center: bool = True,
            seed: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")

        super().__init__(
            params,
            k,
            eps=eps,
            weight_decay=weight_decay,
            history_size=history_size,
            center=center,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self.beta2 = beta2
        self.exp_avg_sq = [
            torch.zeros_like(p, dtype=torch.float32, device=p.device)
            for p in self.params
        ]

    def update_basis(self, loss_closure: Closure) -> tuple[torch.Tensor, torch.Tensor]:
        grads = self._compute_grads(loss_closure)

        for exp_avg_sq, param, grad in zip(self.exp_avg_sq, self.params, grads):
            g = self._grad_or_zero(param, grad)
            exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

        pieces = [v.sqrt().add(self.eps).reciprocal() for v in self.exp_avg_sq]
        return self._set_basis_from_vector(_flatten(pieces))


class AdaptiveLRFullUpdateProjector(_AdaptiveLRProjector):
    """Projector onto covariance directions of full Adam-style updates."""

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            lr: float = 1.0,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            history_size: int | None = None,
            center: bool = True,
            seed: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}.")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")

        super().__init__(
            params,
            k,
            eps=eps,
            weight_decay=weight_decay,
            history_size=history_size,
            center=center,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.step = 0
        self.exp_avg = [
            torch.zeros_like(p, dtype=torch.float32, device=p.device)
            for p in self.params
        ]
        self.exp_avg_sq = [
            torch.zeros_like(p, dtype=torch.float32, device=p.device)
            for p in self.params
        ]

    def update_basis(self, loss_closure: Closure) -> tuple[torch.Tensor, torch.Tensor]:
        grads = self._compute_grads(loss_closure)
        self.step += 1

        for exp_avg, exp_avg_sq, param, grad in zip(
                self.exp_avg,
                self.exp_avg_sq,
                self.params,
                grads,
        ):
            g = self._grad_or_zero(param, grad)
            exp_avg.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

        step_size = self.lr * math.sqrt(1.0 - self.beta2 ** self.step)
        step_size /= 1.0 - self.beta1 ** self.step

        pieces = [
            step_size * m / v.sqrt().add(self.eps)
            for m, v in zip(self.exp_avg, self.exp_avg_sq)
        ]
        return self._set_basis_from_vector(_flatten(pieces))


class AdaptiveLRCoordinateProjector(LowRankBasisProjector):
    """Axis-aligned subspace of the top-k coordinates by largest effective Adam LR.

    Adam's per-coordinate effective learning rate is proportional to
    ``1 / (sqrt(v) + eps)`` where ``v`` is the second-moment EMA of gradients.
    This projector selects the ``k`` coordinates with the largest such effective
    LR (equivalently the smallest ``v``) and projects onto that one-hot coordinate
    subspace -- testing whether training "happens" in the few coordinates Adam
    steps hardest in.

    Note: a coordinate that never receives gradient keeps ``v ~= 0`` and so ranks
    highest by ``1 / sqrt(v)``. With a full-dataset, every-step basis refresh the
    second moment warms up quickly, so truly-dead coordinates are rare; this is
    faithful to the literal "top-r by 1/sqrt(v)" definition.
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            beta2: float = 0.999,
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            seed: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        super().__init__(params, k, device=device, dtype=dtype)

        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.seed = seed
        self.exp_avg_sq = [
            torch.zeros_like(p, dtype=torch.float32, device=p.device)
            for p in self.params
        ]

    def update_basis(self, loss_closure: Closure) -> tuple[torch.Tensor, torch.Tensor]:
        grads = _grads_from_closure(self.params, loss_closure)

        for exp_avg_sq, param, grad in zip(self.exp_avg_sq, self.params, grads):
            g = _grad_or_zero(param, grad, self.weight_decay)
            exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

        # Per-coordinate effective LR is proportional to 1 / (sqrt(v) + eps).
        inv_eff_lr = _flatten(
            [v.sqrt().add(self.eps).reciprocal() for v in self.exp_avg_sq]
        )

        scores, idx = torch.topk(inv_eff_lr, self.k, largest=True)

        basis = torch.zeros(
            self.n_params,
            self.k,
            device=self.device,
            dtype=self.dtype,
        )
        basis[idx.to(self.device), torch.arange(self.k, device=self.device)] = 1.0

        # One-hot columns from distinct indices are already orthonormal.
        self.set_basis(basis, eigvals=scores.detach().cpu(), orthonormalize=False)

        if self.basis is None or self.eigvals is None:
            raise RuntimeError("Adaptive LR coordinate projector failed to build a basis.")

        return self.eigvals, self.basis
