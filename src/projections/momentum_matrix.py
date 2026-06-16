from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import torch

from src.projections.base import (
    BaseProjector,
    ProjectionMode,
    ProjectorInfo,
    _flatten,
    _tree_norm,
    _unflatten_like,
)


MatrixShape = Literal["auto"] | tuple[int, int]
MomentumProjectionType = Literal["two_sided", "tangent", "tangent_rand"]
LayerwiseRankMode = Literal["fixed", "fraction", "energy"]


@dataclass
class _LayerwiseSVDBasis:
    index: int
    matrix_shape: tuple[int, int]
    rank: int
    U: torch.Tensor
    V: torch.Tensor
    singular_values: torch.Tensor


def _auto_matrix_shape(n: int) -> tuple[int, int]:
    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = int(math.ceil(n / rows))
    return rows, cols


def flatten_optimizer_state(params: Sequence[torch.nn.Parameter], optimizer: torch.optim.Optimizer, state_key: str, *,
                            device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    parts: list[torch.Tensor] = []
    found = 0

    for p in params:
        state = optimizer.state.get(p, {})
        value = state.get(state_key)
        if value is None:
            parts.append(torch.zeros(p.numel(), device=device, dtype=dtype))
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"optimizer.state[p][{state_key!r}] must be a tensor, "
                            f"got {type(value).__name__}.")
        if value.shape != p.shape:
            raise ValueError(f"optimizer state {state_key!r} has shape {tuple(value.shape)}, "
                             f"expected parameter shape {tuple(p.shape)}.")
        parts.append(value.detach().reshape(-1).to(device=device, dtype=dtype))
        found += 1

    if not parts:
        raise ValueError("No parameters were selected for optimizer-state flattening.")

    return torch.cat(parts, dim=0), found


class GlobalMomentumSVDProjector(BaseProjector):

    def __init__(self, params: Iterable[torch.nn.Parameter], k: int, *, matrix_shape: MatrixShape = "auto", state_key: str = "momentum_buffer",
                 projection_type: MomentumProjectionType = "two_sided", include_bias: bool = False, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:

        self.all_params = list(params)
        self.trainable_indices = [
            i for i, p in enumerate(self.all_params)
            if p.requires_grad and (include_bias or p.ndim >= 2)
        ]
        self.params = [self.all_params[i] for i in self.trainable_indices]
        if not self.params:
            raise ValueError("No trainable parameters were selected.")

        self.k = int(k)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        self.n_params = int(sum(p.numel() for p in self.params))
        self.device = torch.device(device) if device is not None else self.params[0].device
        self.dtype = dtype if dtype is not None else self.params[0].dtype
        self.matrix_shape = matrix_shape
        self.state_key = state_key
        self.projection_type = projection_type
        self.include_bias = bool(include_bias)

        if self.projection_type not in ("two_sided", "tangent"):
            raise ValueError(
                "projection_type must be 'two_sided' or 'tangent', "
                f"got {projection_type!r}."
            )

        self.rows, self.cols = self._resolve_matrix_shape()
        if self.k > min(self.rows, self.cols):
            raise ValueError(
                f"k={self.k} must be <= min(matrix_shape)={min(self.rows, self.cols)}."
            )

        self.padded_numel = self.rows * self.cols
        self.pad = self.padded_numel - self.n_params
        self.U: torch.Tensor | None = None
        self.V: torch.Tensor | None = None
        self.singular_values: torch.Tensor | None = None
        self.eigvals: torch.Tensor | None = None

    @property
    def is_ready(self) -> bool:
        return self.U is not None and self.V is not None

    @property
    def basis(self) -> None:
        return None

    @property
    def projection_dim(self) -> int | None:
        if not self.is_ready:
            return None
        if self.projection_type == "two_sided":
            return self.k * self.k
        if self.projection_type == "tangent":
            return self.k * (self.rows + self.cols - self.k)
        return None

    def _resolve_matrix_shape(self) -> tuple[int, int]:
        if self.matrix_shape == "auto":
            return _auto_matrix_shape(self.n_params)
        rows, cols = self.matrix_shape
        rows = int(rows)
        cols = int(cols)
        if rows * cols < self.n_params:
            raise ValueError(
                f"matrix_shape {(rows, cols)} has only {rows * cols} entries, "
                f"but selected params have {self.n_params}."
            )
        return rows, cols

    def _pad_flat(self, flat: torch.Tensor) -> torch.Tensor:
        flat = flat.to(device=self.device, dtype=self.dtype)
        if flat.numel() != self.n_params:
            raise ValueError(f"Expected flat vector of length {self.n_params}, got {flat.numel()}.")
        if self.pad == 0:
            return flat
        return torch.cat(
            [flat, flat.new_zeros(self.pad)],
            dim=0,
        )

    def _matrix_from_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return self._pad_flat(flat).reshape(self.rows, self.cols)

    def _flat_from_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.reshape(-1)[: self.n_params]

    def update_basis(self, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None:
        flat_state, found = flatten_optimizer_state(
            self.params,
            optimizer,
            self.state_key,
            device=self.device,
            dtype=self.dtype,
        )

        state_norm = torch.linalg.vector_norm(flat_state)
        if found == 0 or float(state_norm.detach().cpu()) == 0.0:
            self.U = None
            self.V = None
            self.singular_values = None
            self.eigvals = None
            return None

        matrix = self._matrix_from_flat(flat_state)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        self.U = U[:, :self.k].contiguous().detach()
        self.V = Vh[:self.k, :].T.contiguous().detach()
        self.singular_values = S[: self.k].detach().cpu()
        self.eigvals = self.singular_values.to(dtype=torch.float32)

        return self.eigvals, (self.U, self.V)

    def project_flat(self, flat_update: torch.Tensor, mode: ProjectionMode) -> torch.Tensor:
        if mode == "none":
            return flat_update
        if not self.is_ready:
            raise RuntimeError(
                "Momentum SVD basis is empty. Call update_basis(...) "
                "after the optimizer has created momentum state."
            )

        Z = self._matrix_from_flat(flat_update)
        projected = self._project_matrix(Z)

        if mode == "dom":
            out = projected
        elif mode == "bulk":
            out = Z - projected
        else:
            raise ValueError(f"Unknown projection mode: {mode}")

        return self._flat_from_matrix(out)

    def _project_matrix(self, Z: torch.Tensor) -> torch.Tensor:
        if self.U is None or self.V is None:
            raise RuntimeError("Momentum SVD basis is empty.")

        U = self.U.to(device=Z.device, dtype=Z.dtype)
        V = self.V.to(device=Z.device, dtype=Z.dtype)

        left = U @ (U.T @ Z)
        two_sided = left @ V @ V.T
        if self.projection_type == "two_sided":
            return two_sided
        if self.projection_type in ("tangent", "tangent_rand"):
            right = (Z @ V) @ V.T
            return left + right - two_sided
        raise ValueError(f"Unknown projection_type: {self.projection_type!r}")

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if not self.is_ready:
            raise RuntimeError("Momentum SVD basis is empty.")
        Z = self._matrix_from_flat(flat_vec.detach())
        z_norm = float(torch.linalg.vector_norm(Z).detach().cpu())
        if z_norm == 0.0:
            return 0.0
        proj_norm = float(torch.linalg.vector_norm(self._project_matrix(Z)).detach().cpu())
        return proj_norm / z_norm

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

        selected_updates = [update[i].detach() for i in self.trainable_indices]
        flat = _flatten(selected_updates)
        projected_flat = self.project_flat(flat, mode)
        projected_selected = _unflatten_like(projected_flat, self.params)

        out = list(update)
        for idx, val in zip(self.trainable_indices, projected_selected):
            out[idx] = val.to(device=out[idx].device, dtype=out[idx].dtype)
        return tuple(out)

    def info_for(self, raw_update: Sequence[torch.Tensor], projected_update: Sequence[torch.Tensor]) -> ProjectorInfo:
        raw_selected = [raw_update[i].detach() for i in self.trainable_indices]
        proj_selected = [projected_update[i].detach() for i in self.trainable_indices]

        raw_norm = float(_tree_norm(raw_selected).cpu())
        projected_norm = float(_tree_norm(proj_selected).cpu())
        alignment = 0.0 if raw_norm == 0.0 else projected_norm / raw_norm

        return ProjectorInfo(
            raw_norm=raw_norm,
            projected_norm=projected_norm,
            alignment=alignment,
            basis_shape=(self.padded_numel, self.k) if self.is_ready else None,
            eigvals=self.eigvals,
        )


class LayerwiseMomentumSVDProjector(BaseProjector):

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int | None,
            *,
            state_key: str = "momentum_buffer",
            projection_type: MomentumProjectionType = "tangent",
            rank_mode: LayerwiseRankMode = "fixed",
            rank_frac: float | None = 0.05,
            max_rank: int | None = None,
            singular_tol: float = 1e-12,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        self.all_params = list(params)
        self.trainable_indices = [
            i for i, p in enumerate(self.all_params) if p.requires_grad
        ]
        self.params = [self.all_params[i] for i in self.trainable_indices]
        if not self.params:
            raise ValueError("No trainable parameters were passed.")

        self.matrix_indices = [
            i for i in self.trainable_indices if self.all_params[i].ndim >= 2
        ]
        if not self.matrix_indices:
            raise ValueError("No trainable matrix-shaped parameters were selected.")

        self.k = k
        self.state_key = state_key
        self.projection_type = projection_type
        self.rank_mode = rank_mode
        self.rank_frac = 0.05 if rank_frac is None else float(rank_frac)
        self.max_rank = None if max_rank is None else int(max_rank)
        self.singular_tol = float(singular_tol)
        self.device = torch.device(device) if device is not None else self.params[0].device
        self.dtype = dtype if dtype is not None else self.params[0].dtype

        if self.projection_type not in ("two_sided", "tangent", "tangent_rand"):
            raise ValueError(
                "projection_type must be 'two_sided', 'tangent' or 'tangent_rand', "
                f"got {projection_type!r}."
            )
        if self.rank_mode not in ("fixed", "fraction", "energy"):
            raise ValueError(f"Unknown rank_mode: {rank_mode!r}.")
        if self.rank_frac <= 0.0:
            raise ValueError(f"rank_frac must be positive, got {rank_frac}.")
        if self.rank_mode == "energy" and self.rank_frac > 1.0:
            raise ValueError(
                "rank_frac is used as an energy threshold in rank_mode='energy' "
                f"and must be <= 1.0, got {rank_frac}."
            )
        if self.max_rank is not None and self.max_rank <= 0:
            raise ValueError(f"max_rank must be positive, got {max_rank}.")

        self.n_params = int(sum(p.numel() for p in self.params))
        self._basis_by_index: dict[int, _LayerwiseSVDBasis] = {}
        self.eigvals: torch.Tensor | None = None

    @property
    def is_ready(self) -> bool:
        return bool(self._basis_by_index)

    @property
    def basis(self) -> None:
        return None

    @property
    def projection_dim(self) -> int | None:
        return self._total_subspace_dim() if self.is_ready else None

    @staticmethod
    def _matrix_shape_for_param(p: torch.Tensor) -> tuple[int, int]:
        rows = int(p.shape[0])
        cols = int(p.numel() // rows)
        return rows, cols

    @staticmethod
    def _matrix_from_param_like(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _random_orthonormal(self, rows: int, rank: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(
            torch.randn(rows, rank, device=self.device, dtype=self.dtype),
            mode="reduced",
        )
        return q[:, :rank].contiguous()

    def _rank_for_shape(self, rows: int, cols: int) -> int:
        min_dim = min(rows, cols)
        if self.rank_mode == "fixed":
            rank = min(self.k, min_dim)
        elif self.rank_mode == "fraction":
            rank = max(1, int(math.floor(self.rank_frac * min_dim)))
            if self.max_rank is not None:
                rank = min(rank, self.max_rank)
            rank = min(rank, min_dim)
        return max(1, rank)

    def _rank_for_singular_values(self, S: torch.Tensor, rows: int, cols: int) -> int:
        if self.rank_mode != "energy":
            return self._rank_for_shape(rows, cols)

        min_dim = min(rows, cols)
        energy = S.detach().pow(2)
        total = energy.sum()
        if float(total.cpu()) <= 0.0:
            return 0

        cumulative = torch.cumsum(energy, dim=0) / total
        rank = int(torch.searchsorted(cumulative, self.rank_frac, right=False).item()) + 1
        rank = min(rank, min_dim)
        if self.max_rank is not None:
            rank = min(rank, self.max_rank)
        return max(1, rank)

    def update_basis(self, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, dict[int, _LayerwiseSVDBasis]] | None:
        bases: dict[int, _LayerwiseSVDBasis] = {}
        singular_chunks: list[torch.Tensor] = []

        for idx in self.matrix_indices:
            p = self.all_params[idx]
            state = optimizer.state.get(p, {})
            value = state.get(self.state_key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise TypeError(
                    f"optimizer.state[p][{self.state_key!r}] must be a tensor, "
                    f"got {type(value).__name__}."
                )
            if value.shape != p.shape:
                raise ValueError(
                    f"optimizer state {self.state_key!r} has shape {tuple(value.shape)}, "
                    f"expected parameter shape {tuple(p.shape)}."
                )

            rows, cols = self._matrix_shape_for_param(p)
            matrix = value.detach().reshape(rows, cols).to(device=self.device, dtype=self.dtype)
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            if S.numel() == 0:
                continue

            state_norm_sq = float(S.detach().pow(2).sum().cpu())
            if state_norm_sq <= 0.0:
                continue

            requested_rank = self._rank_for_singular_values(S, rows, cols)
            positive = S > self.singular_tol
            n_positive = int(positive.sum().item())
            rank = min(requested_rank, n_positive)
            if rank <= 0:
                continue

            top_s = S[:rank].detach()
            if self.projection_type == "tangent_rand":
                basis_U = self._random_orthonormal(rows, rank)
                basis_V = self._random_orthonormal(cols, rank)
            else:
                basis_U = U[:, :rank].contiguous().detach()
                basis_V = Vh[:rank, :].T.contiguous().detach()

            basis = _LayerwiseSVDBasis(
                index=idx,
                matrix_shape=(rows, cols),
                rank=rank,
                U=basis_U.detach(),
                V=basis_V.detach(),
                singular_values=top_s.cpu(),
            )
            bases[idx] = basis
            singular_chunks.append(top_s.cpu())

        self._basis_by_index = bases
        self.eigvals = (
            torch.cat(singular_chunks).to(dtype=torch.float32)
            if singular_chunks else None
        )

        if not bases:
            return None
        return self.eigvals, self._basis_by_index

    def _project_matrix(self, Z: torch.Tensor, basis: _LayerwiseSVDBasis) -> torch.Tensor:
        U = basis.U.to(device=Z.device, dtype=Z.dtype)
        V = basis.V.to(device=Z.device, dtype=Z.dtype)
        left = U @ (U.T @ Z)
        two_sided = left @ V @ V.T

        if self.projection_type == "two_sided":
            return two_sided
        if self.projection_type in ("tangent", "tangent_rand"):
            right = (Z @ V) @ V.T
            return left + right - two_sided
        raise ValueError(f"Unknown projection_type: {self.projection_type!r}")

    def project_update(self, update: Sequence[torch.Tensor], mode: ProjectionMode) -> tuple[torch.Tensor, ...]:
        if len(update) != len(self.all_params):
            raise ValueError(
                f"Expected update of length {len(self.all_params)}, got {len(update)}."
            )
        if mode == "none":
            return tuple(update)
        if not self.is_ready:
            raise RuntimeError(
                "Layerwise momentum SVD basis is empty. Call "
                "update_basis(...) after momentum state exists."
            )

        out = list(update)
        for idx, basis in self._basis_by_index.items():
            ref = self.all_params[idx]
            Z = update[idx].detach().reshape(basis.matrix_shape).to(
                device=basis.U.device,
                dtype=basis.U.dtype,
            )
            dom = self._project_matrix(Z, basis)
            if mode == "dom":
                projected = dom
            elif mode == "bulk":
                projected = Z - dom
            else:
                raise ValueError(f"Unknown projection mode: {mode}")
            out[idx] = projected.reshape_as(ref).to(device=out[idx].device, dtype=out[idx].dtype)
        return tuple(out)

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if not self.is_ready:
            raise RuntimeError("Layerwise momentum SVD basis is empty.")

        pieces = _unflatten_like(
            flat_vec.detach().to(device=self.device, dtype=self.dtype),
            self.params,
        )
        idx_to_piece = {
            idx: piece
            for idx, piece in zip(self.trainable_indices, pieces)
        }

        denom_sq = flat_vec.detach().pow(2).sum().to(device=self.device, dtype=self.dtype)
        if float(denom_sq.detach().cpu()) == 0.0:
            return 0.0

        num_sq = denom_sq.new_zeros(())
        for idx, basis in self._basis_by_index.items():
            piece = idx_to_piece[idx]
            Z = piece.reshape(basis.matrix_shape)
            dom = self._project_matrix(Z, basis)
            num_sq = num_sq + dom.pow(2).sum()

        return float(torch.sqrt(num_sq / denom_sq).detach().cpu())

    def info_for(
            self,
            raw_update: Sequence[torch.Tensor],
            projected_update: Sequence[torch.Tensor],
    ) -> ProjectorInfo:
        raw_train = [raw_update[i].detach() for i in self.trainable_indices]
        proj_train = [projected_update[i].detach() for i in self.trainable_indices]

        raw_norm = float(_tree_norm(raw_train).cpu())
        projected_norm = float(_tree_norm(proj_train).cpu())
        alignment = 0.0 if raw_norm == 0.0 else projected_norm / raw_norm

        return ProjectorInfo(
            raw_norm=raw_norm,
            projected_norm=projected_norm,
            alignment=alignment,
            basis_shape=(len(self._basis_by_index), self._total_subspace_dim())
            if self.is_ready else None,
            eigvals=self.eigvals,
        )

    def _total_subspace_dim(self) -> int:
        total = 0
        for basis in self._basis_by_index.values():
            rows, cols = basis.matrix_shape
            r = basis.rank
            if self.projection_type == "two_sided":
                total += r * r
            elif self.projection_type in ("tangent", "tangent_rand"):
                total += r * (rows + cols - r)
        return total


def update_momentum_matrix_projector(projector: Any, ctx: Any) -> None:
    if ctx.optimizer is None:
        raise RuntimeError("ProjectorContext.optimizer is None; cannot read momentum state.")
    projector.update_basis(ctx.optimizer)
