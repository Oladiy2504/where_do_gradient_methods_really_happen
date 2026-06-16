from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import warnings
import torch
from torch.func import functional_call, grad, vmap

from src.projections.base import _flatten
from src.projections.eigenspace import MatrixEigenspaceProjector


Batch = Any
LossFn = Callable[[Any, Batch], torch.Tensor]
PerSampleGradMethod = Literal["auto", "vmap", "loop"]
GradientCovarianceSolver = Literal["gram_eigh", "eigsh", "dense_eigh", "cola_lanczos"]


class _FunctionalModel:

    def __init__(
            self,
            base: torch.nn.Module,
            params: Mapping[str, torch.Tensor],
            buffers: Mapping[str, torch.Tensor],
    ) -> None:
        self.base = base
        self.params = params
        self.buffers = buffers
        self.training = base.training

    def __call__(self, *args, **kwargs):
        return functional_call(
            self.base,
            (self.params, self.buffers),
            args,
            kwargs,
        )


def _batch_size(batch: Batch) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for value in batch.values():
            return _batch_size(value)
        return 0
    if isinstance(batch, (tuple, list)):
        return _batch_size(batch[0])
    raise TypeError(f"Cannot determine batch size for type {type(batch).__name__}.")


def _select_one(batch: Batch, idx: int) -> Batch:
    if torch.is_tensor(batch):
        return batch[idx:idx + 1]
    if isinstance(batch, Mapping):
        return type(batch)({k: _select_one(v, idx) for k, v in batch.items()})
    if isinstance(batch, tuple):
        return type(batch)(_select_one(x, idx) for x in batch)
    if isinstance(batch, list):
        return [_select_one(x, idx) for x in batch]
    raise TypeError(f"Cannot slice batch of type {type(batch).__name__}.")


def _slice_batch(batch: Batch, start: int, end: int) -> Batch:
    if torch.is_tensor(batch):
        return batch[start:end]
    if isinstance(batch, Mapping):
        return type(batch)({k: _slice_batch(v, start, end) for k, v in batch.items()})
    if isinstance(batch, tuple):
        return type(batch)(_slice_batch(x, start, end) for x in batch)
    if isinstance(batch, list):
        return [_slice_batch(x, start, end) for x in batch]
    raise TypeError(f"Cannot slice batch of type {type(batch).__name__}.")


def _batch_in_dims(batch: Batch) -> Batch:
    if torch.is_tensor(batch):
        return 0
    if isinstance(batch, Mapping):
        return type(batch)({k: _batch_in_dims(v) for k, v in batch.items()})
    if isinstance(batch, tuple):
        return type(batch)(_batch_in_dims(x) for x in batch)
    if isinstance(batch, list):
        return [_batch_in_dims(x) for x in batch]
    raise TypeError(f"Unsupported batch field type for vmap: {type(batch).__name__}.")


def _unsqueeze_batch_dim(batch: Batch) -> Batch:
    if torch.is_tensor(batch):
        return batch.unsqueeze(0)
    if isinstance(batch, Mapping):
        return type(batch)({k: _unsqueeze_batch_dim(v) for k, v in batch.items()})
    if isinstance(batch, tuple):
        return type(batch)(_unsqueeze_batch_dim(x) for x in batch)
    if isinstance(batch, list):
        return [_unsqueeze_batch_dim(x) for x in batch]
    raise TypeError(f"Unsupported batch field type for vmap: {type(batch).__name__}.")


def _flat_grad_from_loss(
        loss: torch.Tensor,
        params: Sequence[torch.Tensor],
        *,
        retain_graph: bool,
        create_graph: bool = False,
) -> torch.Tensor:
    if loss.ndim != 0:
        loss = loss.mean()

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )

    return _flatten(
        tuple(
            torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(params, grads)
        )
    )


def flat_gradients_from_losses(
        losses: torch.Tensor,
        params: Sequence[torch.Tensor],
) -> torch.Tensor:
    if losses.ndim == 0:
        losses = losses.reshape(1)
    else:
        losses = losses.reshape(-1)

    flat_grads = []
    n_losses = int(losses.numel())

    for i, loss_i in enumerate(losses):
        flat_grads.append(
            _flat_grad_from_loss(
                loss_i,
                params,
                retain_graph=i < n_losses - 1,
                create_graph=False,
            )
        )

    return torch.stack(flat_grads, dim=0)


def flat_gradients_from_loss_fn_loop(
        model: Any,
        batch: Batch,
        params: Sequence[torch.Tensor],
        loss_fn: LossFn,
) -> torch.Tensor:
    n_samples = _batch_size(batch)
    if n_samples <= 0:
        raise ValueError("Cannot build gradient covariance from an empty batch.")

    flat_grads: list[torch.Tensor] = []

    for i in range(n_samples):
        one_batch = _select_one(batch, i)
        with torch.enable_grad():
            loss = loss_fn(model, one_batch)
        flat_grads.append(
            _flat_grad_from_loss(
                loss,
                params,
                retain_graph=False,
                create_graph=False,
            )
        )

    return torch.stack(flat_grads, dim=0)


def flat_gradients_from_loss_fn_vmap(
        model: torch.nn.Module,
        batch: Batch,
        loss_fn: LossFn,
        *,
        chunk_size: int | None = None,
) -> torch.Tensor:
    n_samples = _batch_size(batch)
    if n_samples <= 0:
        raise ValueError("Cannot build gradient covariance from an empty batch.")

    params_dict = dict(model.named_parameters())
    buffers_dict = dict(model.named_buffers())
    param_names = list(params_dict.keys())

    if not param_names:
        raise ValueError("Model has no named parameters to differentiate.")

    def single_loss(
            params: Mapping[str, torch.Tensor],
            buffers: Mapping[str, torch.Tensor],
            one_sample: Batch,
    ) -> torch.Tensor:
        one_batch = _unsqueeze_batch_dim(one_sample)
        fmodel = _FunctionalModel(model, params, buffers)
        loss = loss_fn(fmodel, one_batch)
        if loss.ndim != 0:
            loss = loss.mean()
        return loss

    grad_fn = grad(single_loss, argnums=0)
    in_dims = (None, None, _batch_in_dims(batch))
    vmapped_grad_fn = vmap(grad_fn, in_dims=in_dims)

    chunks: list[torch.Tensor] = []
    if chunk_size is None or chunk_size <= 0:
        ranges = [(0, n_samples)]
    else:
        ranges = [(s, min(s + chunk_size, n_samples)) for s in range(0, n_samples, chunk_size)]

    for start, end in ranges:
        chunk = _slice_batch(batch, start, end)
        grads_dict = vmapped_grad_fn(params_dict, buffers_dict, chunk)
        n_chunk = end - start

        flat_parts: list[torch.Tensor] = []
        for name in param_names:
            g = grads_dict[name]
            flat_parts.append(g.reshape(n_chunk, -1))
        chunks.append(torch.cat(flat_parts, dim=1).detach())

    return torch.cat(chunks, dim=0)


def flat_gradients_from_loss_fn(
        model: Any,
        batch: Batch,
        params: Sequence[torch.Tensor],
        loss_fn: LossFn,
        *,
        method: PerSampleGradMethod = "auto",
        chunk_size: int | None = None,
) -> torch.Tensor:
    if method not in ("auto", "vmap", "loop"):
        raise ValueError(f"Unknown per-sample gradient method: {method!r}.")

    if method in ("auto", "vmap") and isinstance(model, torch.nn.Module):
        try:
            return flat_gradients_from_loss_fn_vmap(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            if method == "vmap":
                raise
            warnings.warn(
                "torch.func vmap per-sample gradients failed; falling back to "
                f"the slower sample-by-sample loop. Original error: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    return flat_gradients_from_loss_fn_loop(
        model=model,
        batch=batch,
        params=params,
        loss_fn=loss_fn,
    )


def empirical_gradient_covariance(
        flat_grads: torch.Tensor,
        *,
        center: bool = True,
        ddof: int = 1,
) -> torch.Tensor:
    if flat_grads.ndim != 2:
        raise ValueError(
            f"flat_grads must have shape [num_samples, num_params], got {tuple(flat_grads.shape)}."
        )

    n_samples = int(flat_grads.shape[0])
    denom = n_samples - ddof
    if denom <= 0:
        raise ValueError(
            f"Need num_samples - ddof > 0, got num_samples={n_samples}, ddof={ddof}."
        )

    X = flat_grads
    if center:
        X = X - X.mean(dim=0, keepdim=True)

    return (X.T @ X) / denom


class GradientCovarianceEigenspaceProjector(MatrixEigenspaceProjector):

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            k: int,
            *,
            center: bool = True,
            ddof: int = 1,
            per_sample_grad_method: PerSampleGradMethod = "auto",
            per_sample_grad_chunk_size: int | None = 16,
            solver: GradientCovarianceSolver = "gram_eigh",
            which: str = "LA",
            tol: float = 1e-3,
            gram_eigval_tol: float = 1e-10,
            maxiter: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            seed: int | None = None,
    ) -> None:
        self.center = center
        self.ddof = ddof
        self.per_sample_grad_method = per_sample_grad_method
        self.per_sample_grad_chunk_size = per_sample_grad_chunk_size
        self.gram_eigval_tol = gram_eigval_tol
        self._centered_flat_grads: torch.Tensor | None = None
        self._denom: int | None = None
        self.mean_grad: torch.Tensor | None = None

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

    def _prepare_flat_grads(self, flat_grads: torch.Tensor) -> None:
        if flat_grads.ndim != 2:
            raise ValueError(
                f"flat_grads must have shape [num_samples, num_params], got {tuple(flat_grads.shape)}."
            )
        if flat_grads.shape[1] != self.n_params:
            raise ValueError(
                f"flat_grads second dimension must be {self.n_params}, got {flat_grads.shape[1]}."
            )

        n_samples = int(flat_grads.shape[0])
        denom = n_samples - self.ddof
        if denom <= 0:
            raise ValueError(
                f"Need num_samples - ddof > 0, got num_samples={n_samples}, ddof={self.ddof}."
            )

        G = flat_grads.detach().to(device=self.device, dtype=self.dtype)

        if self.center:
            mean = G.mean(dim=0, keepdim=True)
            self.mean_grad = mean.squeeze(0).detach().cpu()
            G = G - mean
        else:
            self.mean_grad = None

        self._centered_flat_grads = G.contiguous()
        self._denom = denom

    def _matvec(self, flat_vec: torch.Tensor) -> torch.Tensor:
        if self._centered_flat_grads is None or self._denom is None:
            raise RuntimeError(
                "Gradient samples are not set. Call update_basis(flat_grads=...), "
                "update_basis(losses_closure=...), or update_basis_from_loss_fn(...)."
            )

        X = self._centered_flat_grads
        v = flat_vec.to(device=X.device, dtype=X.dtype)

        return (X.T @ (X @ v)) / self._denom

    def update_basis_from_loss_fn(
            self,
            model: Any,
            batch: Batch,
            loss_fn: LossFn,
            *,
            method: PerSampleGradMethod | None = None,
            chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_grads = flat_gradients_from_loss_fn(
            model=model,
            batch=batch,
            params=self.params,
            loss_fn=loss_fn,
            method=self.per_sample_grad_method if method is None else method,
            chunk_size=self.per_sample_grad_chunk_size if chunk_size is None else chunk_size,
        )
        return self.update_basis(flat_grads=flat_grads)

    def _update_basis_gram_eigh(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._centered_flat_grads is None or self._denom is None:
            raise RuntimeError("Gradient samples are not set.")

        if self.which not in ("LA", "LM"):
            raise ValueError(
                "solver='gram_eigh' is intended for top covariance directions "
                "and supports only which='LA' or which='LM'."
            )

        X = self._centered_flat_grads
        denom = self._denom
        n_samples = int(X.shape[0])

        if self.center and self.k > n_samples - 1:
            raise ValueError(
                f"With center=True, gradient covariance rank is at most "
                f"num_samples - 1 = {n_samples - 1}, but k={self.k}. "
                "Increase basis_subsample or reduce k."
            )
        if not self.center and self.k > n_samples:
            raise ValueError(
                f"With center=False, gradient second-moment rank is at most "
                f"num_samples = {n_samples}, but k={self.k}. "
                "Increase basis_subsample or reduce k."
            )

        K = (X @ X.T) / denom
        K = 0.5 * (K + K.T)
        K_cpu = K.detach().to(device="cpu", dtype=torch.float64)

        vals_cpu, U_cpu = torch.linalg.eigh(K_cpu)

        if self.which == "LA":
            scores = vals_cpu
        else:
            scores = vals_cpu.abs()

        order = torch.argsort(scores, descending=True)
        vals_cpu = vals_cpu[order]
        U_cpu = U_cpu[:, order]

        max_abs = vals_cpu.abs().max().item() if vals_cpu.numel() else 0.0
        tol = max(float(self.gram_eigval_tol), torch.finfo(vals_cpu.dtype).eps * max(1.0, max_abs))
        positive = vals_cpu > tol
        n_positive = int(positive.sum().item())
        if n_positive < self.k:
            raise ValueError(
                f"Only {n_positive} positive Gram eigenvalues above tol={tol:.3e} "
                f"are available, but k={self.k}. Increase basis_subsample, "
                "reduce k, set center=False, or lower gram_eigval_tol."
            )

        eigvals_cpu = vals_cpu[positive][: self.k].contiguous()
        U_cpu = U_cpu[:, positive][:, : self.k].contiguous()

        U = U_cpu.to(device=X.device, dtype=X.dtype)
        eigvals = eigvals_cpu.to(device=X.device, dtype=X.dtype)

        basis = X.T @ U
        scale = torch.sqrt((float(denom) * eigvals).clamp_min(torch.finfo(eigvals.dtype).tiny))
        basis = basis / scale.unsqueeze(0)

        self.set_basis(
            basis.detach(),
            eigvals=eigvals_cpu.to(dtype=torch.float32),
            orthonormalize=False,
        )

        return self.eigvals, self.basis

    def update_basis(
            self,
            losses_closure: Callable[[], torch.Tensor] | None = None,
            *,
            flat_grads: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flat_grads is None:
            if losses_closure is None:
                raise ValueError("Pass either flat_grads or losses_closure.")

            with torch.enable_grad():
                losses = losses_closure()
            flat_grads = flat_gradients_from_losses(losses, self.params)

        self._prepare_flat_grads(flat_grads)

        try:
            if self.solver == "gram_eigh":
                eigvals, basis = self._update_basis_gram_eigh()
            else:
                if self.solver == "dense_eigh":
                    self.matrix = empirical_gradient_covariance(
                        self._centered_flat_grads,
                        center=False,
                        ddof=self.ddof,
                    )

                eigvals, basis = super().update_basis()
        finally:

            self._centered_flat_grads = None
            self._denom = None
            self.matrix = None

        return eigvals, basis


def update_gradient_covariance_projector(projector: Any, ctx: Any) -> None:
    if ctx.basis_batch is None:
        raise RuntimeError("basis_batch is None; cannot build gradient covariance basis.")
    projector.update_basis_from_loss_fn(
        model=ctx.model,
        batch=ctx.basis_batch,
        loss_fn=ctx.task.loss_fn,
    )
