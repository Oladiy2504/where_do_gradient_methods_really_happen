from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch

ProjectionMode = Literal["none", "dom", "bulk"]


@dataclass
class ProjectorInfo:
    raw_norm: float
    projected_norm: float
    alignment: float
    basis_shape: tuple[int, int] | None
    eigvals: torch.Tensor | None = None


def _flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)


def _rademacher(
        size: int | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
) -> torch.Tensor:
    bits = torch.rand(size, device=device, dtype=dtype) < 0.5
    return bits.to(dtype).mul_(2).sub_(1)


def _unflatten_like(flat: torch.Tensor, like: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    pieces = []
    pos = 0

    for ref in like:
        n = ref.numel()
        pieces.append(flat[pos: pos + n].view_as(ref))
        pos += n

    if pos != flat.numel():
        raise ValueError(f"Flat vector has {flat.numel()} entries, consumed {pos}.")

    return tuple(pieces)


def _tree_norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.vector_norm(_flatten([t.detach() for t in tensors]))


class BaseProjector(ABC):
    @abstractmethod
    def update_basis(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def project_update(
            self,
            update: Sequence[torch.Tensor],
            projection: ProjectionMode,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def info_for(
            self,
            raw_update: Sequence[torch.Tensor],
            projected_update: Sequence[torch.Tensor],
    ) -> ProjectorInfo:
        raise NotImplementedError


class LowRankBasisProjector(BaseProjector):
    """
    База для проекторов, которые хранят ортонормированный Q
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        self.all_params = list(params)
        self.trainable_indices = [
            i for i, p in enumerate(self.all_params) if p.requires_grad
        ]
        self.params = [p for p in self.all_params if p.requires_grad]

        if len(self.params) == 0:
            raise ValueError("No trainable parameters were passed.")

        self.k = int(k)
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        self.n_params = int(sum(p.numel() for p in self.params))

        if self.k >= self.n_params:
            raise ValueError(
                f"k={self.k} must be smaller than number of parameters={self.n_params}."
            )

        self.device = torch.device(device) if device is not None else self.params[0].device
        self.dtype = dtype if dtype is not None else self.params[0].dtype

        self.basis: torch.Tensor | None = None
        self.eigvals: torch.Tensor | None = None

    @property
    def is_ready(self) -> bool:
        return self.basis is not None

    def set_basis(
            self,
            basis: torch.Tensor,
            *,
            eigvals: torch.Tensor | None = None,
            orthonormalize: bool = True,
    ) -> None:
        if basis.ndim != 2:
            raise ValueError(f"basis must be 2D, got shape {tuple(basis.shape)}.")

        if basis.shape[0] != self.n_params:
            raise ValueError(
                f"basis has wrong first dimension: {basis.shape[0]}, "
                f"expected {self.n_params}."
            )

        if basis.shape[1] != self.k:
            raise ValueError(
                f"basis has wrong second dimension: {basis.shape[1]}, "
                f"expected k={self.k}."
            )

        basis = basis.to(device=self.device, dtype=self.dtype)

        if orthonormalize:
            basis, _ = torch.linalg.qr(basis, mode="reduced")

        self.basis = basis.detach()
        self.eigvals = None if eigvals is None else eigvals.detach().cpu()

    @abstractmethod
    def update_basis(self, *args, **kwargs):
        raise NotImplementedError

    def chi_k_of(self, flat_vec: torch.Tensor) -> float:
        if self.basis is None:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        v = flat_vec.detach().to(device=self.basis.device, dtype=self.basis.dtype)
        v_norm = float(torch.linalg.vector_norm(v).cpu())
        if v_norm == 0.0:
            return 0.0
        proj_norm = float(torch.linalg.vector_norm(self.basis.T @ v).cpu())
        return proj_norm / v_norm

    def project_flat(self, flat_update: torch.Tensor, mode: ProjectionMode) -> torch.Tensor:
        if mode == "none":
            return flat_update

        if self.basis is None:
            raise RuntimeError("Projector basis is empty. Call update_basis(...) first.")

        u = flat_update.to(device=self.basis.device, dtype=self.basis.dtype)

        dom = self.basis @ (self.basis.T @ u)

        if mode == "dom":
            return dom

        if mode == "bulk":
            return u - dom

        raise ValueError(f"Unknown projection mode: {mode}")

    def project_update(self, update: Sequence[torch.Tensor], mode: ProjectionMode) -> tuple[torch.Tensor, ...]:
        if len(update) != len(self.all_params):
            raise ValueError(
                f"Expected update of length {len(self.all_params)}, got {len(update)}."
            )

        if mode == "none":
            return tuple(update)

        train_updates = [update[i].detach() for i in self.trainable_indices]
        flat = _flatten(train_updates)

        projected_flat = self.project_flat(flat, mode)

        projected_train = _unflatten_like(
            projected_flat,
            [self.all_params[i] for i in self.trainable_indices],
        )

        out = list(update)

        for idx, val in zip(self.trainable_indices, projected_train):
            out[idx] = val.to(device=out[idx].device, dtype=out[idx].dtype)

        return tuple(out)

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
            basis_shape=None if self.basis is None else tuple(self.basis.shape),
            eigvals=self.eigvals,
        )


def project_or_passthrough(
        raw_update: Sequence[torch.Tensor],
        projector: BaseProjector | None,
        projection: ProjectionMode,
) -> tuple[tuple[torch.Tensor, ...], ProjectorInfo]:
    if projector is None or projection == "none":
        flat = _flatten([u.detach() for u in raw_update])
        raw_norm = float(torch.linalg.vector_norm(flat).cpu())
        info = ProjectorInfo(
            raw_norm=raw_norm,
            projected_norm=raw_norm,
            alignment=1.0,
            basis_shape=None,
            eigvals=None,
        )
        return tuple(raw_update), info

    projected = projector.project_update(raw_update, projection)
    info = projector.info_for(raw_update, projected)
    return tuple(projected), info
