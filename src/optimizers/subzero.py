"""SubZero: Zeroth-Order Fine-Tuning of LLMs in Random Subspaces.

Reference:
    Yu, Zhou, Wang, Li, Tian, Huang.
    "Zeroth-Order Fine-Tuning of LLMs in Random Subspaces."
    arXiv:2410.08989, 2025. https://github.com/zimingyy/SubZero

Like MeZO, SubZero estimates a gradient with the SPSA finite-difference rule

    g ~= ((f(theta + eps z_tilde) - f(theta - eps z_tilde)) / (2 eps)) * z_tilde

but instead of drawing ``z_tilde`` from an isotropic Gaussian over the full
parameter space, the perturbation for every weight matrix W_i is drawn from
a layer-wise low-rank subspace,

    Z_tilde_i = U_i Z_i V_i^T,

where U_i and V_i are column-orthogonal matrices obtained from the QR
decomposition of fresh Gaussian draws and Z_i in R^{r x r} is i.i.d.
Gaussian. The subspaces ``(U_i, V_i)`` are refreshed every ``update_freq``
steps (Algorithm 3 in the paper) and reused otherwise.

Parameters with fewer than two dimensions (biases, LayerNorm scales, ...)
fall back to the standard MeZO-style isotropic Gaussian perturbation, which
matches the reference implementation's ``zo_subspace_perturb_parameters``
in ``large_models/trainer.py``.

Randomness is drawn from a private ``torch.Generator`` so the global
PyTorch RNG is left untouched, exactly as in ``mezo.py``. Without an
external projector, the perturbation is regenerated from the same
generator state in every phase and is never materialised as a tuple,
preserving the inference-level memory footprint of the original
algorithm. When a projector is supplied, the full direction tuple must
be materialised so the raw update can be projected.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)


class SubZero(Optimizer):
    """Zeroth-order SGD with layer-wise low-rank perturbations.

    Args:
        params: iterable of parameters or parameter groups.
        lr: learning rate.
        eps: perturbation scale.
        weight_decay: decoupled L2 regularization coefficient.
        seed: optional seed for the private ``torch.Generator``.
        rank: target rank ``r`` of the per-matrix perturbation
            ``Z_tilde_i = U_i Z_i V_i^T``. The effective rank for any
            given parameter is clipped to ``min(rank, m, n)`` where
            ``m, n`` are the rows and columns after reshaping the
            parameter to a 2D matrix (``[fan_out, -1]``).
        update_freq: lazy update interval ``F``. ``U_i`` and ``V_i`` are
            regenerated when ``step_count % update_freq == 0`` and reused
            otherwise.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-6,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int | None = None,
        rank: int = 8,
        update_freq: int = 100,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq: {update_freq}")
        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "rank": rank,
            "update_freq": update_freq,
        }
        super().__init__(params, defaults)

        device = self._all_params()[0].device
        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self._step_count: int = 0
        self.last_info: dict[str, object] = {}

    def _all_params(self) -> list[torch.Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    @staticmethod
    def _matrix_shape(p: torch.Tensor) -> tuple[int, int] | None:
        """Return ``(m, n)`` for a 2D-reshapable param, else ``None``."""
        if p.ndim < 2:
            return None
        m = p.shape[0]
        n = p.numel() // m
        return m, n

    def _effective_rank(self, p: torch.Tensor, rank: int) -> int:
        shape = self._matrix_shape(p)
        if shape is None:
            return 0
        m, n = shape
        return max(1, min(rank, m, n))

    def _generate_uv(self, p: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """QR-decompose two fresh Gaussian draws into column-orthogonal U, V."""
        shape = self._matrix_shape(p)
        assert shape is not None
        m, n = shape
        r_eff = self._effective_rank(p, rank)
        r1 = torch.randn(
            (m, r_eff),
            device=p.device,
            dtype=p.dtype,
            generator=self._generator,
        )
        r2 = torch.randn(
            (n, r_eff),
            device=p.device,
            dtype=p.dtype,
            generator=self._generator,
        )
        u, _ = torch.linalg.qr(r1)
        v, _ = torch.linalg.qr(r2)
        return u.contiguous(), v.contiguous()

    def _maybe_refresh_uv(self, rank: int, update_freq: int) -> bool:
        """Refresh per-matrix bases at the lazy-update boundary."""
        refresh = (self._step_count % update_freq) == 0
        if not refresh:
            return False
        for p in self._all_params():
            if not p.requires_grad:
                continue
            if self._matrix_shape(p) is None:
                continue
            u, v = self._generate_uv(p, rank)
            state = self.state[p]
            state["U"] = u
            state["V"] = v
        return True

    def _sample_ztilde(self, p: torch.Tensor, rank: int) -> torch.Tensor:
        """Draw the next perturbation for parameter ``p``.

        Matrix-shaped params: sample ``Z in R^{r x r}`` and return
        ``(U @ Z @ V^T) * sqrt(numel(W) / numel(Z))`` reshaped to
        ``p.shape``. The ``sqrt(numel/numel)`` factor (taken from the
        reference implementation) keeps the perturbation's element-wise
        variance comparable to an isotropic Gaussian on the full matrix.

        Non-matrix params: fall back to ``N(0, I)`` of ``p.shape``.
        """
        shape = self._matrix_shape(p)
        if shape is None:
            return torch.randn(
                p.shape,
                device=p.device,
                dtype=p.dtype,
                generator=self._generator,
            )

        state = self.state[p]
        u: torch.Tensor = state["U"]
        v: torch.Tensor = state["V"]
        r_eff = u.shape[1]
        z = torch.randn(
            (r_eff, r_eff),
            device=p.device,
            dtype=p.dtype,
            generator=self._generator,
        )
        z_tilde = u @ z @ v.T
        scale = math.sqrt(p.numel() / max(z.numel(), 1))
        z_tilde = z_tilde * scale
        return z_tilde.view(p.shape)

    @torch.no_grad()
    def _shift_inplace(
        self, gen_state: torch.Tensor, alpha: float, rank: int
    ) -> None:
        """Re-sample ``Z_tilde`` from ``gen_state`` and add ``alpha * Z_tilde`` to each param."""
        self._generator.set_state(gen_state)
        for p in self._all_params():
            if not p.requires_grad:
                continue
            z_tilde = self._sample_ztilde(p, rank)
            p.data.add_(z_tilde, alpha=alpha)

    @torch.no_grad()
    def _materialize_ztilde(
        self, gen_state: torch.Tensor, rank: int
    ) -> tuple[torch.Tensor, ...]:
        """Materialise ``Z_tilde`` aligned with ``self._all_params()``."""
        self._generator.set_state(gen_state)
        out: list[torch.Tensor] = []
        for p in self._all_params():
            if not p.requires_grad:
                out.append(torch.zeros_like(p))
            else:
                out.append(self._sample_ztilde(p, rank))
        return tuple(out)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        *,
        projector: BaseProjector | None = None,
        projection: ProjectionMode = "none",
    ) -> torch.Tensor:
        """Run one SubZero step. ``closure`` must return a scalar loss tensor."""
        if closure is None:
            raise ValueError("SubZero requires a closure that returns a scalar loss.")
        if projection != "none" and projector is None:
            raise ValueError("projection is not 'none', but projector is None.")

        defaults = self.param_groups[0]
        eps = defaults["eps"]
        rank = defaults["rank"]
        update_freq = defaults["update_freq"]

        bases_refreshed = self._maybe_refresh_uv(rank, update_freq)

        gen_state = self._generator.get_state()
        use_projection = projector is not None and projection != "none"

        if use_projection:
            z_tilde = self._materialize_ztilde(gen_state, rank)
            for p, zi in zip(self._all_params(), z_tilde):
                if p.requires_grad:
                    p.data.add_(zi, alpha=eps)
        else:
            z_tilde = None
            self._shift_inplace(gen_state, alpha=+eps, rank=rank)

        with torch.enable_grad():
            loss_plus = closure()
        loss_plus_val = float(loss_plus.detach())

        if use_projection:
            for p, zi in zip(self._all_params(), z_tilde):
                if p.requires_grad:
                    p.data.add_(zi, alpha=-2.0 * eps)
        else:
            self._shift_inplace(gen_state, alpha=-2.0 * eps, rank=rank)

        with torch.enable_grad():
            loss_minus = closure()
        loss_minus_val = float(loss_minus.detach())

        if use_projection:
            for p, zi in zip(self._all_params(), z_tilde):
                if p.requires_grad:
                    p.data.add_(zi, alpha=+eps)
        else:
            self._shift_inplace(gen_state, alpha=+eps, rank=rank)

        coeff = (loss_plus_val - loss_minus_val) / (2.0 * eps)

        if use_projection:
            raw_update = tuple(coeff * zi for zi in z_tilde)
            update, info = project_or_passthrough(raw_update, projector, projection)

            self.last_info = {
                "projection": projection,
                "mezo_coeff": coeff,
                "raw_update_norm": info.raw_norm,
                "projected_update_norm": info.projected_norm,
                "alignment": info.alignment,
                "eigvals": info.eigvals,
                "rank": rank,
                "update_freq": update_freq,
                "bases_refreshed": bases_refreshed,
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
            for group in self.param_groups:
                lr = group["lr"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    zi = self._sample_ztilde(p, rank)
                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(zi, alpha=-lr * coeff)

            self.last_info = {
                "projection": "none",
                "mezo_coeff": coeff,
                "raw_update_norm": None,
                "projected_update_norm": None,
                "alignment": 1.0,
                "eigvals": None,
                "rank": rank,
                "update_freq": update_freq,
                "bases_refreshed": bases_refreshed,
            }

        self._step_count += 1

        return torch.tensor(
            0.5 * (loss_plus_val + loss_minus_val),
            device=self._all_params()[0].device,
        )
