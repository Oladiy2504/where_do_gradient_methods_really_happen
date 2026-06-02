from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

from src.projections.base import _flatten, _unflatten_like
from src.projections.eigenspace import MatrixEigenspaceProjector


def hessian_vector_product(
        loss_closure: Callable[[], torch.Tensor],
        params: Sequence[torch.Tensor],
        vector: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Compute H v without explicitly forming H."""

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
    """Second backward of a pre-computed differentiable gradient against `vector`.

    `grads` must come from `autograd.grad(loss, params, create_graph=True)` —
    i.e. the first backward must still hold its graph. Pass retain_graph=True
    when the same `grads` will be reused for further HVPs (e.g. inside an
    ARPACK matvec loop), False on the last call.
    """
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


class HessianEigenspaceProjector(MatrixEigenspaceProjector):
    """Projector onto top-k Hessian eigenspace.

    This is just MatrixEigenspaceProjector where the matrix-vector product
    is defined by HVP.
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

        # Initialize base class with a placeholder matvec.
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

        # retain_graph=True: ARPACK will keep asking for matvecs against the
        # same cached first backward.
        hvp_tree = _hvp_from_grads(
            self._cached_grads,
            self.params,
            vec_tree,
            retain_graph=True,
        )

        return _flatten(hvp_tree).to(device=self.device, dtype=self.dtype)

    def update_basis(self, loss_closure: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Run forward + first backward ONCE per basis refresh. The autograd
        # graph behind `grads` stays alive as long as `self._cached_grads`
        # holds the references, so every ARPACK matvec only pays for the
        # second backward.
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
            # Drop refs so the autograd graph (and held activations) is freed.
            self._cached_grads = None

        return eigvals, basis
