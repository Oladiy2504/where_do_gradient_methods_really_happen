from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch

from src.projections.base import LowRankBasisProjector

from scipy.sparse.linalg import LinearOperator, eigsh

Eigensolver = Literal["eigsh", "dense_eigh", "cola_lanczos"]
Which = Literal["LA", "LM", "SA", "SM"]


class MatrixEigenspaceProjector(LowRankBasisProjector):
    """Projector onto top-k eigenspace of a symmetric matrix/operator.

    The matrix can be provided either explicitly or as a matrix-vector product.

    Args:
        params:
            Model parameters defining the ambient optimization space.
        k:
            Subspace dimension.
        matrix:
            Optional explicit matrix A of shape [num_params, num_params].
            Only suitable for small models.
        matvec:
            Optional callable v -> A v.
            This is the preferred mode for large models.
        solver:
            "eigsh" for scipy ARPACK over a LinearOperator (CPU, fp64),
            "dense_eigh" for explicit dense symmetric matrix (USE ONLY IF MATRIX IS SMALL!),
            "cola_lanczos" for cola-ml Lanczos staying on `self.device` in `self.dtype`
              (no H2D/D2H per matvec, no fp32->fp64 promotion). Recommended for HVP.
        which:
            For eigsh:
              - "LA": largest algebraic eigenvalues;
              - "LM": largest magnitude eigenvalues.
    """

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

        # For safety: enforce symmetry numerically.
        A = 0.5 * (A + A.T)

        eigvals, eigvecs = torch.linalg.eigh(A)

        order = torch.argsort(eigvals, descending=True)[: self.k]

        top_vals = eigvals[order].detach().cpu()
        top_vecs = eigvecs[:, order].detach()

        # eigh returns orthonormal eigenvectors; QR would only rotate columns
        # within degenerate eigenspaces and break the eigval<->column mapping.
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

        # Warm-start: when a basis from a previous call exists, hand ARPACK its
        # top column as the initial Lanczos vector. On consecutive refreshes
        # (e.g. every-step Hessian top-k for Song et al. replication) this
        # dramatically accelerates convergence because successive Hessians
        # share most of their dominant subspace. We fall back to a seeded
        # random vector on cold start (no basis yet) for reproducibility, and
        # to ARPACK's own RNG if neither is available.
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

        # eigsh returns orthonormal eigenvectors; same reasoning as dense_eigh.
        self.set_basis(basis, eigvals=eigvals, orthonormalize=False)

        return self.eigvals, self.basis

    def _update_basis_cola_lanczos(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy import: keep src.projections importable when cola-ml is absent.
        import cola
        from cola.ops import LinearOperator as ColaLO
        from cola.linalg.decompositions.lanczos import lanczos

        n = self.n_params

        def matmat(X: torch.Tensor) -> torch.Tensor:
            # cola's matmat contract is 2D (n, b). Lanczos sends b=1 in practice,
            # so the column loop is a no-op wrapper. MUST NOT be wrapped in
            # torch.no_grad(): the HVP-backed _apply_matvec relies on the cached
            # autograd graph from HessianEigenspaceProjector._cached_grads.
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

        # Low-level call returns Krylov basis Q (n, m) and tridiagonal T (m, m)
        # directly, so we can run the final dense symmetric solve ourselves and
        # keep full control over `which` semantics (cola's high-level `eig` only
        # supports LM/SM, not LA/SA).
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

        # T is (m, m) with m <= max_iters (~40-100), so the dense symmetric
        # solve is trivial and we run it on CPU — torch.linalg.eigh is not
        # implemented on MPS yet (pytorch#141287).
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

        # Q is unitary (Lanczos) and S is orthogonal (eigh), so top_vecs columns
        # are orthonormal — skip QR for the same reason as the eigsh path.
        self.set_basis(top_vecs, eigvals=top_vals, orthonormalize=False)

        return self.eigvals, self.basis

    def _cola_start_vector(self, n: int) -> torch.Tensor | None:
        # Warm start: reuse the dominant column from the previous basis. For
        # the every-step Hessian refresh protocol successive Hessians share
        # most of their top subspace, so this slashes the matvec count.
        if self.basis is not None:
            return self.basis[:, 0].detach().to(device=self.device, dtype=self.dtype)
        if self.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            return torch.randn(n, generator=gen, device=self.device, dtype=self.dtype)
        return None

    def _cola_max_iters(self, n: int) -> int:
        # cola.lanczos has no implicit restart and effectively ignores `tol`
        # for HVP-style operators (subdiagonal beta_i stays large while top-k
        # already converged), so `max_iters` acts as a hard matvec budget.
        # Empirically `4*k` Krylov steps (floored at 40) match eigsh tol=1e-4
        # numerics on MLP3-MNIST (rel_err ~1e-6, subspace alignment ~1.0)
        # both cold and warm. Going larger only wastes HVPs and Krylov
        # memory `(1, n, max_iters+2)` allocated upfront.
        if self.maxiter is not None:
            return int(self.maxiter)
        return min(max(4 * self.k, 40), n - 1)
