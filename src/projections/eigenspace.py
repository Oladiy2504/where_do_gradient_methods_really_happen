from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch

from src.projections.base import LowRankBasisProjector

from scipy.sparse.linalg import LinearOperator, eigsh

Eigensolver = Literal["eigsh", "dense_eigh", "cola_lanczos"]
Which = Literal["LA", "LM", "SA", "SM"]


class MatrixEigenspaceProjector(LowRankBasisProjector):

    def __init__(
            self,
            params,
            k: int,
            *,
            matrix: torch.Tensor | None = None,
            matvec: Callable[[torch.Tensor], torch.Tensor] | None = None,
            solver: Eigensolver = "eigsh",
            which: Which = "LA",
            tol: float = 1e-3,
            maxiter: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            seed: int | None = None,
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)

        if matrix is None and matvec is None:
            raise ValueError("Either matrix or matvec must be provided.")

        if matrix is not None and matvec is not None:
            raise ValueError("Pass only one of matrix or matvec, not both.")

        self.matrix = matrix
        self.matvec = matvec
        self.solver = solver
        self.which = which
        self.tol = tol
        self.maxiter = maxiter
        self.seed = seed

        if self.matrix is not None:
            if self.matrix.shape != (self.n_params, self.n_params):
                raise ValueError(
                    f"matrix must have shape {(self.n_params, self.n_params)}, "
                    f"got {tuple(self.matrix.shape)}."
                )

    def _apply_matvec(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=self.device, dtype=self.dtype)

        if self.matrix is not None:
            A = self.matrix.to(device=self.device, dtype=self.dtype)
            return A @ v

        if self.matvec is None:
            raise RuntimeError("matvec is not defined.")

        return self.matvec(v).to(device=self.device, dtype=self.dtype)

    def update_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.solver == "dense_eigh":
            return self._update_basis_dense_eigh()

        if self.solver == "eigsh":
            return self._update_basis_eigsh()

        if self.solver == "cola_lanczos":
            return self._update_basis_cola_lanczos()

        raise ValueError(f"Unknown solver: {self.solver}")

    def _update_basis_dense_eigh(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.matrix is None:
            raise ValueError("dense_eigh requires an explicit matrix.")

        A = self.matrix.to(device=self.device, dtype=self.dtype)

        A = 0.5 * (A + A.T)

        eigvals, eigvecs = torch.linalg.eigh(A)

        order = torch.argsort(eigvals, descending=True)[: self.k]

        top_vals = eigvals[order].detach().cpu()
        top_vecs = eigvecs[:, order].detach()

        self.set_basis(top_vecs, eigvals=top_vals, orthonormalize=False)

        return self.eigvals, self.basis

    def _update_basis_eigsh(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_params

        def scipy_matvec(x_np: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.asarray(x_np)).to(
                device=self.device,
                dtype=self.dtype,
            )

            y = self._apply_matvec(x)

            return y.detach().cpu().numpy().astype(np.float64, copy=False)

        operator = LinearOperator(
            shape=(n, n),
            matvec=scipy_matvec,
            dtype=np.float64,
        )

        v0 = None
        if self.basis is not None:
            v0 = self.basis[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        elif self.seed is not None:
            rng = np.random.default_rng(self.seed)
            v0 = rng.standard_normal(n).astype(np.float64)

        vals_np, vecs_np = eigsh(
            operator,
            k=self.k,
            which=self.which,
            tol=self.tol,
            maxiter=self.maxiter,
            return_eigenvectors=True,
            v0=v0,
        )

        eigvals = torch.from_numpy(vals_np).to(dtype=torch.float32)
        basis = torch.from_numpy(vecs_np).to(device=self.device, dtype=self.dtype)

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        basis = basis[:, order]

        self.set_basis(basis, eigvals=eigvals, orthonormalize=False)

        return self.eigvals, self.basis

    def _update_basis_cola_lanczos(self) -> tuple[torch.Tensor, torch.Tensor]:

        import cola
        from cola.ops import LinearOperator as ColaLO
        from cola.linalg.decompositions.lanczos import lanczos

        n = self.n_params

        def matmat(X: torch.Tensor) -> torch.Tensor:

            cols = [self._apply_matvec(X[:, j]) for j in range(X.shape[1])]
            return torch.stack(cols, dim=1)

        op = ColaLO(
            dtype=self.dtype,
            shape=(n, n),
            matmat=matmat,
            annotations={cola.SelfAdjoint},
        )

        start = self._cola_start_vector(n)
        max_iters = self._cola_max_iters(n)

        Q_lo, T_lo, _info = lanczos(
            op,
            start_vector=start,
            max_iters=max_iters,
            tol=self.tol,
        )
        Q = Q_lo.to_dense().to(device=self.device, dtype=self.dtype)
        T = T_lo.to_dense().to(device=self.device, dtype=self.dtype)

        if Q.ndim == 3:
            Q = Q[0]
        if T.ndim == 3:
            T = T[0]

        T_cpu = T.detach().to(device="cpu", dtype=torch.float32)
        vals_cpu, S_cpu = torch.linalg.eigh(0.5 * (T_cpu + T_cpu.T))
        vals = vals_cpu.to(device=self.device, dtype=self.dtype)
        S = S_cpu.to(device=self.device, dtype=self.dtype)

        if self.which in ("LA", "LM"):
            scores = vals if self.which == "LA" else vals.abs()
            order = torch.argsort(scores, descending=True)[: self.k]
        elif self.which in ("SA", "SM"):
            scores = vals if self.which == "SA" else vals.abs()
            order = torch.argsort(scores, descending=False)[: self.k]
        else:
            raise ValueError(f"Unknown which={self.which!r}")

        top_vals = vals[order].detach().to(device="cpu", dtype=torch.float32)
        top_vecs = (Q @ S[:, order]).detach()

        self.set_basis(top_vecs, eigvals=top_vals, orthonormalize=False)

        return self.eigvals, self.basis

    def _cola_start_vector(self, n: int) -> torch.Tensor | None:

        if self.basis is not None:
            return self.basis[:, 0].detach().to(device=self.device, dtype=self.dtype)
        if self.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            return torch.randn(n, generator=gen, device=self.device, dtype=self.dtype)
        return None

    def _cola_max_iters(self, n: int) -> int:

        if self.maxiter is not None:
            return int(self.maxiter)
        return min(max(4 * self.k, 40), n - 1)
