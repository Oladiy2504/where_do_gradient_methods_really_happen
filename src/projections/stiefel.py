from __future__ import annotations

from typing import Callable, Iterable, Sequence

import math
import torch

from src.projections.base import LowRankBasisProjector, _flatten
from src.optimizers.muon import _cans_explicit3, _delta_orthogonalization



class StiefelProjector(LowRankBasisProjector):
    """
    Learnable projector UU^T, where U lies on the Stiefel manifold:
    
    U.T @ U = I
    
    U is trained online to capture recent graduents:
    
    maximize 0.5 * || U.T @ g ||^2
    """
    
    def __init__(self,
        params: Iterable[torch.nn.Parameter],
        k: int,
        *,
        lr: float = 1e-2,
        seed: int | None = None,
        normalize_vectors: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        eps: float = 1e-12,
        retraction_method: str = "cayley",
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)

        self.lr = lr
        self.seed = seed
        self.retraction_method = retraction_method
        self.normalize_vectors = normalize_vectors
        self.eps = eps
        self.num_updates = 0

        self._init_basis()
        
    def _init_basis(self) -> None:
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
        
        U = torch.randn(
            self.n_params,
            self.k,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        
        self.set_basis(U, eigvals=None, orthonormalize=True)
        
    @staticmethod
    def _sym(A: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A + A.T)


    def _cayley_retraction_woodbury(
        self,
        X: torch.Tensor,
        direction: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Cayley retraction via Woodbury formula.
        """
        
        # L = [alpha M, X], shape: n x 2k
        # R = [X^T;
        #      alpha (M^T X X^T - M^T)], shape: 2k x n
        # Y = X + 0.5 * alpha * M
        # X_next = Y + 1/2 L (I - 1/2 R L)^(-1) R Y
        
        M = direction
        L = torch.cat([alpha * M, X], dim=1)
        MTX = M.T @ X
        R_bottom = alpha * (MTX @ X.T - M.T)
        R = torch.cat([X.T, R_bottom], dim=0)
        Y = X + 0.5 * alpha * M

        small_matrix = torch.eye(
            2 * self.k,
            device=X.device,
            dtype=X.dtype,
        ) - 0.5 * (R @ L)

        rhs = R @ Y
        correction = torch.linalg.solve(small_matrix, rhs)

        X_next = Y + 0.5 * (L @ correction)

        return X_next
    
    def _cans_retraction(self, X: torch.Tensor, direction: torch.Tensor, alpha: float, steps: int = 5, cans_preprocess_steps: int = 4, cans_delta: float = 0.99) -> torch.Tensor:
        """
        Retraction via CANS iteration.
        """
        Y = X + alpha * direction

        one_norm = torch.linalg.norm(Y, ord=1)
        inf_norm = torch.linalg.norm(Y, ord=float("inf"))
        scale = torch.rsqrt((one_norm * inf_norm).clamp_min(self.eps))
        Y = Y * scale
        
        pre_coeffs, _ = _delta_orthogonalization(
            n=cans_preprocess_steps,
            delta=cans_delta,
        )
        for c1, c3, _ in pre_coeffs:
            YTY = Y.T @ Y
            Y = c1 * Y + c3 * (Y @ YTY)
            
        left, right = 1.0 - cans_delta, 1.0 + cans_delta
        
        for _ in range(steps):
            c1, c3, err = _cans_explicit3(left, right)
            YTY = Y.T @ Y
            Y = c1 * Y + c3 * (Y @ YTY)
            left, right = 1.0 - err, 1.0 + err
            
        return Y

    def update_from_flat_vector(self, flat_vec: torch.Tensor) -> tuple[None, torch.Tensor]:
        if self.basis is None:
            self._init_basis()

        U = self.basis.to(device=self.device, dtype=self.dtype)
        g = flat_vec.detach().to(device=self.device, dtype=self.dtype)

        if g.numel() != self.n_params:
            raise ValueError(
                f"Expected vector of length {self.n_params}, got {g.numel()}."
            )

        g_norm = torch.linalg.vector_norm(g)
        if g_norm <= self.eps:
            return None, U

        if self.normalize_vectors:
            g = g / g_norm.clamp_min(self.eps)

        a = U.T @ g
        riem_grad = g[:, None] * a[None, :] - U @ (a[:, None] @ a[None, :])

        with torch.no_grad():
            if self.retraction_method == "cayley":
                U_new = self._cayley_retraction_woodbury(U, riem_grad, self.lr)
            elif self.retraction_method == "cans":
                U_new = self._cans_retraction(U, riem_grad, self.lr)
            self.set_basis(U_new, eigvals=None, orthonormalize=False)
            self.num_updates += 1

        return None, self.basis

    def update_basis(
        self,
        loss_closure: Callable[[], torch.Tensor] | None = None,
        vector: Sequence[torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor]:

        if vector is not None:
            if torch.is_tensor(vector):
                flat = vector.reshape(-1)
            else:
                flat = _flatten([v.detach() for v in vector])

            return self.update_from_flat_vector(flat)

        if loss_closure is None:
            return None, self.basis

        with torch.enable_grad():
            loss = loss_closure()
            if loss.ndim != 0:
                loss = loss.mean()

            grads = torch.autograd.grad(
                loss,
                self.params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

        grad_tree = [
            torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(self.params, grads)
        ]

        flat_grad = _flatten(grad_tree)

        return self.update_from_flat_vector(flat_grad)