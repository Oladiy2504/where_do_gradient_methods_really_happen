from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh

from src.projections.base import (
    LowRankBasisProjector,
    ProjectionMode,
    _rademacher,
    _unflatten_like,
)
from src.projections.hessian import _hvp_from_grads, _hvp_from_grads_batched


Hvp2D = Callable[[torch.Tensor], torch.Tensor]

BatchedHvp2D = Callable[[torch.Tensor], torch.Tensor]

_MAX_HVP_BATCH = 64
_PROBE_ELEM_BUDGET = 1 << 24
_DENSE_DIM_MAX = 64


def _col_chunk(r: int, m: int, n: int) -> int:
    """
    Сколько столбцов mode-operator можно прогнать одним batched HVP
    """
    by_batch = max(1, _MAX_HVP_BATCH // max(1, r))
    by_elems = max(1, _PROBE_ELEM_BUDGET // max(1, r * m * n))
    return max(1, min(by_batch, by_elems))


def _order_topk(vals: torch.Tensor, which: str, r: int) -> torch.Tensor:
    scores = vals if which == "LA" else vals.abs()
    return torch.argsort(scores, descending=True)[:r]


def _spectral_norm_2d(
        op: Hvp2D,
        shape: tuple[int, int],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        iters: int = 20,
) -> float:
    z = _rademacher(shape, device=device, dtype=dtype)
    z = z / (z.norm() + 1e-30)
    spec = 0.0
    for _ in range(iters):
        w = op(z)
        nrm = float(w.norm().cpu())
        if nrm == 0.0:
            return 0.0
        spec = nrm
        z = w / nrm
    return spec


@dataclass
class TwoSidedFactor:
    U: torch.Tensor
    V: torch.Tensor
    fan_out: int
    eigvals: torch.Tensor | None = None
    L_pos: torch.Tensor | None = None
    L_neg: torch.Tensor | None = None
    R_pos: torch.Tensor | None = None
    R_neg: torch.Tensor | None = None


def _topk_eig_of_operator(
        matmat: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        r: int,
        *,
        which: str,
        warm: torch.Tensor | None,
        tol: float,
        maxiter: int | None,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    r = min(r, dim)
    use_dense = dim <= _DENSE_DIM_MAX or r >= dim - 1

    if not use_dense:
        if device.type == "cuda":
            try:
                return _cola_topk(
                    matmat, dim, r, which=which, warm=warm,
                    maxiter=maxiter, tol=tol, device=device, dtype=dtype,
                )
            except Exception:
                pass
        try:
            return _arpack_topk(
                matmat, dim, r, which=which, warm=warm,
                tol=tol, maxiter=maxiter, device=device, dtype=dtype,
            )
        except ArpackError:
            pass

    A = matmat(torch.eye(dim, device=device, dtype=dtype))
    A = 0.5 * (A + A.T)
    A_cpu = A.detach().to(device="cpu", dtype=torch.float32)
    vals_cpu, vecs_cpu = torch.linalg.eigh(A_cpu)
    order = _order_topk(vals_cpu, which, r)
    return vecs_cpu[:, order].to(device=device, dtype=dtype), vals_cpu[order]


def _arpack_topk(
        matmat: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        r: int,
        *,
        which: str,
        warm: torch.Tensor | None,
        tol: float,
        maxiter: int | None,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:

    def scipy_matvec(x_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_np)).to(device=device, dtype=dtype).reshape(dim, 1)
        return matmat(x).reshape(dim).detach().cpu().numpy().astype(np.float64, copy=False)

    operator = LinearOperator(shape=(dim, dim), matvec=scipy_matvec, dtype=np.float64)

    v0 = None
    if warm is not None and warm.shape[0] == dim and warm.numel() >= dim:
        v0 = warm.reshape(dim, -1)[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)

    vals_np, vecs_np = eigsh(
        operator,
        k=r,
        which=which,
        tol=tol,
        maxiter=maxiter,
        return_eigenvectors=True,
        v0=v0,
    )
    vals = torch.from_numpy(vals_np).to(dtype=torch.float32)
    vecs = torch.from_numpy(vecs_np).to(device=device, dtype=dtype)
    order = _order_topk(vals, which, r)
    return vecs[:, order], vals[order].cpu()


def _cola_topk(
        matmat: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        r: int,
        *,
        which: str,
        warm: torch.Tensor | None,
        maxiter: int | None,
        tol: float,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    import cola
    from cola.ops import LinearOperator as ColaLO
    from cola.linalg.decompositions.lanczos import lanczos

    op = ColaLO(dtype=dtype, shape=(dim, dim), matmat=matmat, annotations={cola.SelfAdjoint})

    start = None
    if warm is not None and warm.shape[0] == dim and warm.numel() >= dim:
        start = warm.reshape(dim, -1)[:, 0].detach().to(device=device, dtype=dtype)
    max_iters = int(maxiter) if maxiter is not None else min(max(4 * r, 40), dim - 1)

    Q_lo, T_lo, _info = lanczos(op, start_vector=start, max_iters=max_iters, tol=tol)
    Q = Q_lo.to_dense().to(device=device, dtype=dtype)
    T = T_lo.to_dense().to(device=device, dtype=dtype)
    if Q.ndim == 3:
        Q = Q[0]
    if T.ndim == 3:
        T = T[0]

    T_cpu = T.detach().to(device="cpu", dtype=torch.float32)
    vals_cpu, S_cpu = torch.linalg.eigh(0.5 * (T_cpu + T_cpu.T))
    order = _order_topk(vals_cpu, which, r)
    top_vecs = (Q @ S_cpu.to(device=device, dtype=dtype)[:, order]).detach()
    return top_vecs, vals_cpu[order]


def _rand_orth(
        dim: int,
        r: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
) -> torch.Tensor:
    g = torch.randn(dim, r, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(g, mode="reduced")
    return q.to(device=device, dtype=dtype)


def _make_left_mode_op(hvp2d_b: BatchedHvp2D, V: torch.Tensor, m: int, n: int):
    r = V.shape[1]
    chunk = _col_chunk(r, m, n)

    def op(X: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in range(0, X.shape[1], chunk):
            Xc = X[:, s:s + chunk]
            k = Xc.shape[1]
            probes = torch.einsum("mk,nb->kbmn", Xc, V).reshape(k * r, m, n)
            hp = hvp2d_b(probes).reshape(k, r, m, n)
            outs.append(torch.einsum("kbmn,nb->mk", hp, V))
        return torch.cat(outs, dim=1)

    return op


def _make_right_mode_op(hvp2d_b: BatchedHvp2D, U: torch.Tensor, m: int, n: int):
    r = U.shape[1]
    chunk = _col_chunk(r, m, n)

    def op(Y: torch.Tensor) -> torch.Tensor:
        outs = []
        for s in range(0, Y.shape[1], chunk):
            Yc = Y[:, s:s + chunk]
            k = Yc.shape[1]
            probes = torch.einsum("ma,nk->kamn", U, Yc).reshape(k * r, m, n)
            hp = hvp2d_b(probes).reshape(k, r, m, n)
            outs.append(torch.einsum("kamn,ma->nk", hp, U))
        return torch.cat(outs, dim=1)

    return op


def hooi(
        hvp2d_b: BatchedHvp2D,
        m: int,
        n: int,
        r: int,
        *,
        sweeps: int = 2,
        which: str = "LA",
        tol: float = 1e-4,
        maxiter: int | None = None,
        u_init: torch.Tensor | None = None,
        v_init: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = min(r, m, n)
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32

    if v_init is not None and tuple(v_init.shape) == (n, r):
        V = v_init.to(device=device, dtype=dtype)
    else:
        V = _rand_orth(n, r, device=device, dtype=dtype, generator=generator)

    U = u_init.to(device=device, dtype=dtype) if (u_init is not None and tuple(u_init.shape) == (m, r)) else None
    eig_u = torch.zeros(r)

    for _ in range(sweeps):
        m_op = _make_left_mode_op(hvp2d_b, V, m, n)
        U, eig_u = _topk_eig_of_operator(
            m_op, m, r, which=which, warm=U, tol=tol, maxiter=maxiter, device=device, dtype=dtype
        )

        n_op = _make_right_mode_op(hvp2d_b, U, m, n)
        V, _ = _topk_eig_of_operator(
            n_op, n, r, which=which, warm=V, tol=tol, maxiter=maxiter, device=device, dtype=dtype
        )

    return U, V, eig_u


class TwoSidedBasisProjector(LowRankBasisProjector):
    def __init__(
            self,
            params,
            k: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)
        self._factors: list[TwoSidedFactor | None] = [None] * len(self.params)
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @staticmethod
    def _as_2d(t: torch.Tensor, fan_out: int) -> torch.Tensor:
        return t.reshape(fan_out, -1)

    @staticmethod
    def _dom_2d(d2d: torch.Tensor, f: TwoSidedFactor) -> torch.Tensor:
        U = f.U.to(device=d2d.device, dtype=d2d.dtype)
        V = f.V.to(device=d2d.device, dtype=d2d.dtype)
        return U @ (U.T @ d2d @ V) @ V.T

    def set_two_sided_basis(self, factors: Sequence[TwoSidedFactor | None]) -> None:
        if len(factors) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} factors, got {len(factors)}.")
        self._factors = list(factors)
        eig_pieces = [
            f.eigvals for f in self._factors if f is not None and f.eigvals is not None
        ]
        self.eigvals = torch.cat([e.reshape(-1) for e in eig_pieces]).detach().cpu() if eig_pieces else None
        self._ready = True

    def project_update(
            self,
            update: Sequence[torch.Tensor],
            mode: ProjectionMode,
    ) -> tuple[torch.Tensor, ...]:
        if len(update) != len(self.all_params):
            raise ValueError(
                f"Expected update of length {len(self.all_params)}, got {len(update)}."
            )
        if mode == "none":
            return tuple(update)
        if not self._ready:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        out = list(update)
        for j, idx in enumerate(self.trainable_indices):
            f = self._factors[j]
            orig = update[idx]
            if f is None:
                out[idx] = torch.zeros_like(orig) if mode == "dom" else orig
                continue
            u = orig.detach().to(device=self.device, dtype=self.dtype)
            d2d = self._as_2d(u, f.fan_out)
            dom = self._dom_2d(d2d, f).reshape(orig.shape)
            val = dom if mode == "dom" else (u - dom)
            out[idx] = val.to(device=orig.device, dtype=orig.dtype)
        return tuple(out)

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if not self._ready:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        v = flat_vec.detach().to(device=self.device, dtype=self.dtype)
        v_norm = float(torch.linalg.vector_norm(v).cpu())
        if v_norm == 0.0:
            return 0.0

        parts = _unflatten_like(v, self.params)
        dom_sq = 0.0
        for j, part in enumerate(parts):
            f = self._factors[j]
            if f is None:
                continue
            dom = self._dom_2d(self._as_2d(part, f.fan_out), f)
            dom_sq += float((dom * dom).sum().cpu())
        return (dom_sq ** 0.5) / v_norm


class SpectralHessianProjector(TwoSidedBasisProjector):
    def __init__(
            self,
            params,
            k: int,
            *,
            hooi_sweeps: int = 2,
            which: str = "LA",
            tol: float = 1e-4,
            inner_maxiter: int | None = None,
            seed: int | None = 0,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)
        if hooi_sweeps < 1:
            raise ValueError(f"hooi_sweeps must be >= 1, got {hooi_sweeps}.")
        if which not in ("LA", "LM"):
            raise ValueError(f"which must be 'LA' or 'LM', got {which!r}.")
        self.hooi_sweeps = int(hooi_sweeps)
        self.which = which
        self.tol = tol
        self.inner_maxiter = inner_maxiter
        self.seed = seed
        self._cached_grads: tuple[torch.Tensor | None, ...] | None = None

        self._batched_hvp_ok = True

    def update_basis(self, loss_closure: Callable[[], torch.Tensor]) -> tuple[torch.Tensor | None, None]:

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
            factors: list[TwoSidedFactor | None] = []
            for j, p in enumerate(self.params):
                if p.ndim < 2 or grads[j] is None:
                    factors.append(None)
                    continue
                factors.append(self._hooi_layer(j, p))
            self.set_two_sided_basis(factors)
        finally:
            self._cached_grads = None

        return self.eigvals, None

    def _euclidean_hvp2d(self, j: int, param: torch.Tensor, fan_out: int, n_in: int) -> Hvp2D:

        def hvp2d(d2d: torch.Tensor) -> torch.Tensor:
            probe = d2d.reshape(param.shape).to(device=param.device, dtype=param.dtype)
            hv = _hvp_from_grads(
                [self._cached_grads[j]], [param], [probe], retain_graph=True
            )[0]
            return hv.reshape(fan_out, n_in).to(device=self.device, dtype=self.dtype)

        return hvp2d

    def _euclidean_hvp2d_batched(
            self, j: int, param: torch.Tensor, fan_out: int, n_in: int
    ) -> BatchedHvp2D:

        def hvp2d_b(probes2d: torch.Tensor) -> torch.Tensor:
            b = probes2d.shape[0]
            if self._batched_hvp_ok:
                try:
                    probes = probes2d.reshape(b, *param.shape).to(
                        device=param.device, dtype=param.dtype
                    )
                    hv = _hvp_from_grads_batched(
                        self._cached_grads[j], param, probes, retain_graph=True
                    )
                    return hv.reshape(b, fan_out, n_in).to(
                        device=self.device, dtype=self.dtype
                    )
                except RuntimeError:
                    self._batched_hvp_ok = False
            base = self._euclidean_hvp2d(j, param, fan_out, n_in)
            return torch.stack([base(probes2d[i]) for i in range(b)], dim=0)

        return hvp2d_b

    def _hooi_layer(self, j: int, param: torch.Tensor) -> TwoSidedFactor:
        fan_out = param.shape[0]
        n_in = param.numel() // fan_out
        r = min(self.k, fan_out, n_in)

        hvp2d_b = self._euclidean_hvp2d_batched(j, param, fan_out, n_in)

        prev = self._factors[j]
        gen = torch.Generator().manual_seed((self.seed or 0) * 1_000_003 + j)
        U, V, eig_u = hooi(
            hvp2d_b,
            fan_out,
            n_in,
            r,
            sweeps=self.hooi_sweeps,
            which=self.which,
            tol=self.tol,
            maxiter=self.inner_maxiter,
            u_init=None if prev is None else prev.U,
            v_init=None if prev is None else prev.V,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        return TwoSidedFactor(U=U, V=V, fan_out=fan_out, eigvals=eig_u)

    def _layer_operator(
            self, j: int, param: torch.Tensor, fan_out: int, n_in: int
    ) -> Hvp2D:
        return self._euclidean_hvp2d(j, param, fan_out, n_in)

    def estimate_stable_rank(
            self,
            loss_closure: Callable[[], torch.Tensor],
            *,
            n_probes: int = 16,
            power_iters: int = 20,
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
            frob_sq_total = 0.0
            spec_max = 0.0
            for j, p in enumerate(self.params):
                if p.ndim < 2 or grads[j] is None:
                    continue
                fan_out = p.shape[0]
                n_in = p.numel() // fan_out
                op = self._layer_operator(j, p, fan_out, n_in)
                acc = 0.0
                for _ in range(n_probes):
                    z2d = _rademacher(
                        (fan_out, n_in), device=self.device, dtype=self.dtype
                    )
                    hz = op(z2d)
                    acc += float((hz * hz).sum().cpu())
                frob_sq_total += acc / n_probes
                spec_max = max(
                    spec_max,
                    _spectral_norm_2d(
                        op, (fan_out, n_in),
                        device=self.device, dtype=self.dtype, iters=power_iters,
                    ),
                )
        finally:
            self._cached_grads = None

        return 0.0 if spec_max == 0.0 else frob_sq_total / (spec_max * spec_max)


def _sym_metric_factors(
        g2d: torch.Tensor,
        eps: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    solve_device = torch.device("cpu") if dev.type == "mps" else dev
    g = g2d.detach().to(device=solve_device, dtype=torch.float32)

    def _powers(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam, q = torch.linalg.eigh(gram)
        lam = lam.clamp_min(0.0)
        lam_max = float(lam.max())
        floor = eps * lam_max if lam_max > 0.0 else 1.0
        dvals = lam.clamp_min(floor)
        quarter = dvals.pow(0.25)
        pos = (q * quarter) @ q.T
        neg = (q * quarter.reciprocal()) @ q.T
        return pos, neg

    l_pos, l_neg = _powers(g @ g.T)
    r_pos, r_neg = _powers(g.T @ g)

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=dtype)

    return _to(l_pos), _to(l_neg), _to(r_pos), _to(r_neg)


class MuonMetricHessianProjector(SpectralHessianProjector):
    def __init__(
            self,
            params,
            k: int,
            *,
            eps: float = 0.1,
            whiten_projection: bool = True,
            hooi_sweeps: int = 2,
            which: str = "LA",
            tol: float = 1e-4,
            inner_maxiter: int | None = None,
            seed: int | None = 0,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            params,
            k,
            hooi_sweeps=hooi_sweeps,
            which=which,
            tol=tol,
            inner_maxiter=inner_maxiter,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        self.eps = float(eps)
        self.whiten_projection = bool(whiten_projection)

    def _hooi_layer(self, j: int, param: torch.Tensor) -> TwoSidedFactor:
        fan_out = param.shape[0]
        n_in = param.numel() // fan_out
        r = min(self.k, fan_out, n_in)

        g2d = self._cached_grads[j].reshape(fan_out, n_in)
        l_pos, l_neg, r_pos, r_neg = _sym_metric_factors(
            g2d, self.eps, device=self.device, dtype=self.dtype
        )

        base_hvp2d_b = self._euclidean_hvp2d_batched(j, param, fan_out, n_in)

        def whitened_hvp2d_b(d2d_b: torch.Tensor) -> torch.Tensor:

            hv = base_hvp2d_b(l_neg @ d2d_b @ r_neg)
            return l_neg @ hv @ r_neg

        prev = self._factors[j]
        gen = torch.Generator().manual_seed((self.seed or 0) * 1_000_003 + j)
        U, V, eig_u = hooi(
            whitened_hvp2d_b,
            fan_out,
            n_in,
            r,
            sweeps=self.hooi_sweeps,
            which=self.which,
            tol=self.tol,
            maxiter=self.inner_maxiter,
            u_init=None if prev is None else prev.U,
            v_init=None if prev is None else prev.V,
            device=self.device,
            dtype=self.dtype,
            generator=gen,
        )
        if not self.whiten_projection:

            return TwoSidedFactor(U=U, V=V, fan_out=fan_out, eigvals=eig_u)
        return TwoSidedFactor(
            U=U,
            V=V,
            fan_out=fan_out,
            eigvals=eig_u,
            L_pos=l_pos,
            L_neg=l_neg,
            R_pos=r_pos,
            R_neg=r_neg,
        )

    def _layer_operator(
            self, j: int, param: torch.Tensor, fan_out: int, n_in: int
    ) -> Hvp2D:
        g2d = self._cached_grads[j].reshape(fan_out, n_in)
        _, l_neg, _, r_neg = _sym_metric_factors(
            g2d, self.eps, device=self.device, dtype=self.dtype
        )
        base_hvp2d = self._euclidean_hvp2d(j, param, fan_out, n_in)

        def whitened_hvp2d(d2d: torch.Tensor) -> torch.Tensor:
            hv = base_hvp2d(l_neg @ d2d @ r_neg)
            return l_neg @ hv @ r_neg

        return whitened_hvp2d

    @staticmethod
    def _dom_2d(d2d: torch.Tensor, f: TwoSidedFactor) -> torch.Tensor:
        if f.L_neg is None:
            return TwoSidedBasisProjector._dom_2d(d2d, f)
        dev, dt = d2d.device, d2d.dtype
        U = f.U.to(device=dev, dtype=dt)
        V = f.V.to(device=dev, dtype=dt)
        l_pos = f.L_pos.to(device=dev, dtype=dt)
        l_neg = f.L_neg.to(device=dev, dtype=dt)
        r_pos = f.R_pos.to(device=dev, dtype=dt)
        r_neg = f.R_neg.to(device=dev, dtype=dt)
        y = l_pos @ d2d @ r_pos
        y = U @ (U.T @ y @ V) @ V.T
        return l_neg @ y @ r_neg

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if not self._ready:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        v = flat_vec.detach().to(device=self.device, dtype=self.dtype)
        parts = _unflatten_like(v, self.params)
        num = 0.0
        den = 0.0
        for jj, part in enumerate(parts):
            f = self._factors[jj]
            if f is None:
                den += float((part * part).sum().cpu())
                continue
            d2d = self._as_2d(part, f.fan_out)
            if f.L_pos is None:
                y = d2d
            else:
                l_pos = f.L_pos.to(device=d2d.device, dtype=d2d.dtype)
                r_pos = f.R_pos.to(device=d2d.device, dtype=d2d.dtype)
                y = l_pos @ d2d @ r_pos
            U = f.U.to(device=d2d.device, dtype=d2d.dtype)
            V = f.V.to(device=d2d.device, dtype=d2d.dtype)
            proj = U.T @ y @ V
            num += float((proj * proj).sum().cpu())
            den += float((y * y).sum().cpu())
        if den == 0.0:
            return 0.0
        return (num / den) ** 0.5
