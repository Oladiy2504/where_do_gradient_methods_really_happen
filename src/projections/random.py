from __future__ import annotations

import torch

from src.projections.base import LowRankBasisProjector


class RandomSubspaceProjector(LowRankBasisProjector):

    def __init__(
        self,
        params,
        k: int,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(params, k, device=device, dtype=dtype)
        self.seed = seed

    def update_basis(self) -> tuple[None, torch.Tensor]:
        if self.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            basis = torch.randn(
                self.n_params,
                self.k,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            basis = torch.randn(
                self.n_params,
                self.k,
                device=self.device,
                dtype=self.dtype,
            )

        self.set_basis(basis, eigvals=None, orthonormalize=True)

        return None, self.basis
