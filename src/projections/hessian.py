from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import torch

from src.projections.base import _flatten, _rademacher, _unflatten_like
from src.projections.eigenspace import MatrixEigenspaceProjector


def hessian_vector_product(
        loss_closure: Callable[[], torch.Tensor],
        params: Sequence[torch.Tensor],
        vector: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if len(params) != len(vector):
        raise ValueError("params and vector must have the same length.")

    with torch.enable_grad():
        loss = loss_closure()

        if loss.ndim != 0:
            loss = loss.mean()

        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

    return _hvp_from_grads(grads, params, vector, retain_graph=False)


def _hvp_from_grads(
        grads: Sequence[torch.Tensor | None],
        params: Sequence[torch.Tensor],
        vector: Sequence[torch.Tensor],
        *,
        retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    
    if len(params) != len(vector):
        raise ValueError("params and vector must have the same length.")

    dot_terms = [(g * v).sum() for g, v in zip(grads, vector) if g is not None]

    if not dot_terms:
        return tuple(torch.zeros_like(p) for p in params)

    grad_dot_vec = torch.stack(dot_terms).sum()

    hvps = torch.autograd.grad(
        grad_dot_vec,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tuple(
        torch.zeros_like(p) if hv is None else hv.detach()
        for p, hv in zip(params, hvps)
    )


def _hvp_from_grads_batched(
        grad: torch.Tensor,
        param: torch.Tensor,
        probes: torch.Tensor,
        *,
        retain_graph: bool,
) -> torch.Tensor:
    hv = torch.autograd.grad(
        grad,
        param,
        grad_outputs=probes,
        retain_graph=retain_graph,
        is_grads_batched=True,
        allow_unused=True,
    )[0]
    if hv is None:
        return torch.zeros_like(probes)
    return hv.detach()


class HessianEigenspaceProjector(MatrixEigenspaceProjector):
    """
    Top-k eigenspace гессиана, где matvec задается через HVP
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            solver: str = "eigsh",
            which: str = "LA",
            tol: float = 1e-3,
            maxiter: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            seed: int | None = None,
    ) -> None:
        self._cached_grads: tuple[torch.Tensor | None, ...] | None = None

        super().__init__(
            params=params,
            k=k,
            matvec=self._matvec,
            solver=solver,
            which=which,
            tol=tol,
            maxiter=maxiter,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    def _matvec(self, flat_vec: torch.Tensor) -> torch.Tensor:
        if self._cached_grads is None:
            raise RuntimeError(
                "Cached first-order grads are not set. Call update_basis(loss_closure)."
            )

        flat_vec = flat_vec.to(
            device=self.params[0].device,
            dtype=self.params[0].dtype,
        )

        vec_tree = _unflatten_like(flat_vec, self.params)

        hvp_tree = _hvp_from_grads(
            self._cached_grads,
            self.params,
            vec_tree,
            retain_graph=True,
        )

        return _flatten(hvp_tree).to(device=self.device, dtype=self.dtype)

    def update_basis(self, loss_closure: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Первый backward держим в кеше: eigensolver дальше вызывает только HVP
        with torch.enable_grad():
            loss = loss_closure()

            if loss.ndim != 0:
                loss = loss.mean()

            grads = torch.autograd.grad(
                loss,
                self.params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

        self._cached_grads = grads

        try:
            eigvals, basis = super().update_basis()
        finally:
            self._cached_grads = None

        return eigvals, basis

    def estimate_stable_rank(
            self,
            loss_closure: Callable[[], torch.Tensor],
            *,
            n_probes: int = 16,
    ) -> float:
        if n_probes <= 0:
            raise ValueError(f"n_probes must be positive, got {n_probes}.")

        with torch.enable_grad():
            loss = loss_closure()
            if loss.ndim != 0:
                loss = loss.mean()
            grads = torch.autograd.grad(
                loss,
                self.params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

        self._cached_grads = grads
        try:
            acc = 0.0
            for _ in range(n_probes):
                z = _rademacher(self.n_params, device=self.device, dtype=self.dtype)
                hz = self._matvec(z)
                acc += float((hz * hz).sum().cpu())
            frob_sq = acc / n_probes
        finally:
            self._cached_grads = None

        eig = self.eigvals
        if eig is None or eig.numel() == 0:
            return 0.0
        lam_sq = float(eig.abs().max()) ** 2
        return 0.0 if lam_sq == 0.0 else frob_sq / lam_sq


class AdaptiveHessianEigenspaceProjector(HessianEigenspaceProjector):
    """
    Тот же Hessian top-k, но в диагональной Adam-подобной геометрии
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            beta2: float = 0.999,
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            solver: str = "eigsh",
            which: str = "LA",
            tol: float = 1e-3,
            maxiter: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            seed: int | None = None,
    ) -> None:
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        super().__init__(
            params,
            k,
            solver=solver,
            which=which,
            tol=tol,
            maxiter=maxiter,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.exp_avg_sq = [
            torch.zeros_like(p, dtype=torch.float32, device=p.device)
            for p in self.params
        ]
        self._precond_step = 0

        self._precond_sqrt: torch.Tensor | None = None
        self._precond_inv_sqrt: torch.Tensor | None = None

    def _refresh_preconditioner(
            self,
            grads: Sequence[torch.Tensor | None],
    ) -> None:
        """
        Обновляет EMA второго момента и диагональный предобуславливатель
        """
        for exp_avg_sq, param, grad in zip(self.exp_avg_sq, self.params, grads):
            g = torch.zeros_like(param) if grad is None else grad
            g = g.detach().to(device=param.device, dtype=torch.float32)
            if self.weight_decay != 0.0:
                g = g.add(param.detach().to(dtype=torch.float32), alpha=self.weight_decay)
            exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

        self._precond_step += 1
        bias_correction2 = 1.0 - self.beta2 ** self._precond_step
        denom = _flatten(
            [(v / bias_correction2).sqrt().add(self.eps) for v in self.exp_avg_sq]
        )
        denom = denom.to(device=self.device, dtype=self.dtype)
        self._precond_sqrt = denom.sqrt()
        self._precond_inv_sqrt = denom.rsqrt()

    def _matvec(self, flat_vec: torch.Tensor) -> torch.Tensor:
        if self._precond_inv_sqrt is None:
            raise RuntimeError(
                "Preconditioner is not set. Call update_basis(loss_closure)."
            )
        d = self._precond_inv_sqrt.to(device=flat_vec.device, dtype=flat_vec.dtype)

        return super()._matvec(flat_vec * d) * self._precond_inv_sqrt

    def update_basis(
            self,
            loss_closure: Callable[[], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.enable_grad():
            loss = loss_closure()

            if loss.ndim != 0:
                loss = loss.mean()

            grads = torch.autograd.grad(
                loss,
                self.params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

        self._cached_grads = grads
        self._refresh_preconditioner(grads)

        try:
            eigvals, basis = MatrixEigenspaceProjector.update_basis(self)
        finally:
            self._cached_grads = None

        return eigvals, basis

    def project_flat(self, flat_update: torch.Tensor, mode) -> torch.Tensor:
        if mode == "none":
            return flat_update

        if self.basis is None or self._precond_sqrt is None:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        u = flat_update.to(device=self.basis.device, dtype=self.basis.dtype)
        s = self._precond_sqrt
        d = self._precond_inv_sqrt

        dom = d * (self.basis @ (self.basis.T @ (s * u)))

        if mode == "dom":
            return dom

        if mode == "bulk":
            return u - dom

        raise ValueError(f"Unknown projection mode: {mode}")

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if self.basis is None or self._precond_sqrt is None:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        v = flat_vec.detach().to(device=self.basis.device, dtype=self.basis.dtype)
        w = self._precond_sqrt * v
        w_norm = float(torch.linalg.vector_norm(w).cpu())
        if w_norm == 0.0:
            return 0.0
        proj_norm = float(torch.linalg.vector_norm(self.basis.T @ w).cpu())
        return proj_norm / w_norm

    def precond_diagnostics(self) -> dict[str, float] | None:
        if self._precond_step == 0:
            return None
        bias_correction2 = 1.0 - self.beta2 ** self._precond_step
        sv = torch.cat(
            [(v / bias_correction2).sqrt().flatten() for v in self.exp_avg_sq]
        ).float()

        cap = 1 << 24
        if sv.numel() > cap:
            sv = sv[:: (sv.numel() // (1 << 22)) + 1]
        qs = torch.quantile(
            sv, torch.tensor([0.01, 0.5, 0.99], device=sv.device, dtype=sv.dtype)
        )
        return {
            "sqrt_vhat_p1": float(qs[0]),
            "sqrt_vhat_p50": float(qs[1]),
            "sqrt_vhat_p99": float(qs[2]),
            "sqrt_vhat_min": float(sv.min()),
            "eps": float(self.eps)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "exp_avg_sq": [v.detach().cpu().clone() for v in self.exp_avg_sq],
            "precond_step": self._precond_step
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for dst, src in zip(self.exp_avg_sq, state["exp_avg_sq"]):
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        self._precond_step = int(state["precond_step"])
