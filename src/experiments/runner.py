"""Инфраструктура запуска экспериментов без привязки к конкретной модели."""

from __future__ import annotations

import copy
import inspect
import json
import os
import random
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.func import functional_call
from torch.nn.modules.batchnorm import _BatchNorm

from src.models.utils import seed_everything
from src.projections.base import _flatten

ProjectionMode = Literal["none", "dom", "bulk"]
OptimizerKind = Literal["first_order", "mezo", "forward_gradient"]
ProjectorUpdateKind = Literal["no_args", "loss_closure", "custom"]

FORCE_SWITCH_SENTINEL = ".force_switch"

Batch = Any
LossFn = Callable[[Any, Batch], torch.Tensor]
MetricsFn = Callable[[torch.nn.Module, Batch], Mapping[str, float]]
BatchToDeviceFn = Callable[[Batch, torch.device, torch.dtype | None], Batch]
ProjectorUpdateFn = Callable[[Any, "ProjectorContext"], None]


def cycle(loader: Iterable[Batch]):
    while True:
        for batch in loader:
            yield batch


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.numel() != 1:
            return None
        return float(x.detach().cpu())
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _chi_k_from_alignment(optimizer: Any, projection: ProjectionMode) -> float | None:
    info = getattr(optimizer, "last_info", None)
    alignment = info.get("alignment") if isinstance(info, Mapping) else None
    alignment = _to_float(alignment)
    if alignment is None:
        return None
    a = min(1.0, max(0.0, alignment))
    if projection == "dom":
        return a
    if projection == "bulk":
        return float((1.0 - a * a) ** 0.5)
    return None


def _batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for v in batch.values():
            return _batch_size(v)
        return 0
    if isinstance(batch, (tuple, list)):
        return _batch_size(batch[0])
    raise TypeError(f"Cannot determine batch size for type {type(batch).__name__}.")


def _slice_batch(batch: Any, n: int) -> Any:
    if torch.is_tensor(batch):
        return batch[:n]
    if isinstance(batch, Mapping):
        return type(batch)({k: _slice_batch(v, n) for k, v in batch.items()})
    if isinstance(batch, tuple):
        return type(batch)(_slice_batch(x, n) for x in batch)
    if isinstance(batch, list):
        return [_slice_batch(x, n) for x in batch]
    raise TypeError(f"Cannot slice batch of type {type(batch).__name__}.")


def _jsonable(x: Any) -> Any:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x)
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


@dataclass
class TaskSpec:
    name: str
    model_factory: Callable[[], torch.nn.Module]
    train_loader: Iterable[Batch]
    loss_fn: LossFn
    metrics_fn: MetricsFn | None = None
    batch_to_device: BatchToDeviceFn = None
    basis_loader: Iterable[Batch] | None = None


@dataclass
class OptimizerSpec:
    name: str
    cls: type | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: OptimizerKind = "first_order"
    factory: Callable[[Sequence[torch.nn.Parameter]], torch.optim.Optimizer] | None = None

    def build(self, params: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if self.factory is not None:
            return self.factory(params)
        if self.cls is None:
            raise ValueError(f"OptimizerSpec({self.name!r}) has neither cls nor factory.")
        return self.cls(params, **self.kwargs)


@dataclass
class ProjectorSpec:
    """
    Описание проектора и расписания обновления его базиса
    """

    name: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    modes: Sequence[ProjectionMode] = ("dom", "bulk")
    update_kind: ProjectorUpdateKind = "no_args"
    update_before_train: bool = True
    update_every_steps: int | None = None
    update_fn: ProjectorUpdateFn | None = None
    # True означает basis по всему train set, дороже по памяти.
    basis_full_dataset: bool = False
    basis_subsample: int | None = None
    # Switch начинает с raw-оптимизатора и включает dom/bulk по EMA chi_k.
    switch_on_alignment_ema: float | None = None
    switch_on_step: int | None = None

    def build(self, params: Sequence[torch.nn.Parameter]) -> Any:
        return self.cls(params, **self.kwargs)


@dataclass
class RunnerConfig:
    steps: int
    device: str | torch.device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype: torch.dtype | None = torch.float32
    seed: int = 42
    log_every: int = 50
    include_baseline: bool = True
    fail_fast: bool = True
    save_dir: str | Path | None = None
    keep_models: bool = False
    show_progress: bool = True
    # В статье EMA для chi_k использует alpha=0.9
    chi_ema_factor: float = 0.9
    chi_k_full_batch_every: int | None = None
    compile_model: bool = False
    compile_mode: str | None = None
    log_top_eigvals: int | None = None
    stable_rank_probes: int | None = None
    # SWA хранится в side buffer: текущая траектория оптимизации не меняется
    swa_from_step: int | None = None
    frozen_bulk: bool = False


@dataclass
class ProjectorContext:
    task: TaskSpec
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    optimizer_spec: OptimizerSpec
    projector_spec: ProjectorSpec
    projection: ProjectionMode
    step: int
    device: torch.device
    dtype: torch.dtype | None
    basis_batch: Batch | None
    loss_closure: Callable[[], torch.Tensor]


@dataclass
class SwitchCheckpoint:
    step: int
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    projector_state: dict[str, Any] | None
    chi_ema: float | None
    torch_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None
    numpy_rng_state: Any
    python_rng_state: Any
    paired_dom_run_id: str | None = None


@dataclass
class PlanEntry:
    opt_spec: OptimizerSpec
    proj_spec: ProjectorSpec | None
    modes: tuple[ProjectionMode, ...]


@dataclass
class RunResult:
    run_name: str
    optimizer: str
    projector: str
    projection: ProjectionMode
    history: list[dict[str, Any]]
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None
    model: torch.nn.Module | None = None

    @property
    def final_loss(self) -> float | None:
        if not self.history:
            return None
        return self.history[-1].get("loss")


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

    def __call__(self, *args, **kwargs):
        return functional_call(self.base, (self.params, self.buffers), args, kwargs)


class ExperimentRunner:

    def __init__(
            self,
            task: TaskSpec,
            optimizers: Sequence[OptimizerSpec],
            projectors: Sequence[ProjectorSpec] | None,
            config: RunnerConfig,
    ) -> None:
        self.task = task
        self.optimizers = list(optimizers)
        self.projectors = list(projectors or [])
        self.config = config
        self.device = torch.device(config.device)

        self.save_dir = None if config.save_dir is None else Path(config.save_dir)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[RunResult]:
        results: list[RunResult] = []
        plan = self._build_plan()

        for entry in plan:

            seed = self.config.seed
            if entry.modes == ("dom", "bulk"):
                phase_results = self._run_dom_then_bulk(entry, seed)
            else:
                phase_results = [
                    self._run_one(entry.opt_spec, entry.proj_spec, mode, seed)
                    for mode in entry.modes
                ]

            for result in phase_results:
                results.append(result)
                if self.save_dir is not None:
                    self._save_result(result)
                if result.status == "failed" and self.config.fail_fast:
                    raise RuntimeError(result.error)

        if self.save_dir is not None:
            self._save_summary(results)

        return results

    def _build_plan(self) -> list[PlanEntry]:
        plan: list[PlanEntry] = []
        seen: set[tuple[str, str, tuple[ProjectionMode, ...]]] = set()

        def add(
            opt_spec: OptimizerSpec,
            proj_spec: ProjectorSpec | None,
            modes: tuple[ProjectionMode, ...],
        ) -> None:
            projector_name = "none" if proj_spec is None else proj_spec.name
            key = (opt_spec.name, projector_name, modes)
            if key not in seen:
                seen.add(key)
                plan.append(PlanEntry(opt_spec=opt_spec, proj_spec=proj_spec, modes=modes))

        for opt_spec in self.optimizers:
            if self.config.include_baseline:
                add(opt_spec, None, ("none",))

            for proj_spec in self.projectors:
                subspace_modes = tuple(m for m in proj_spec.modes if m != "none")
                has_none = any(m == "none" for m in proj_spec.modes)
                if has_none:
                    add(opt_spec, None, ("none",))

                if subspace_modes == ("dom", "bulk") or subspace_modes == ("bulk", "dom"):
                    add(opt_spec, proj_spec, ("dom", "bulk"))
                else:
                    for mode in subspace_modes:
                        add(opt_spec, proj_spec, (mode,))

        return plan

    def _run_one(
            self,
            opt_spec: OptimizerSpec,
            proj_spec: ProjectorSpec | None,
            projection: ProjectionMode,
            seed: int,
            *,
            on_switch: Callable[[SwitchCheckpoint], None] | None = None,
            resume_from: SwitchCheckpoint | None = None,
            run_name_override: str | None = None,
    ) -> RunResult:
        projector_name = "none" if proj_spec is None else proj_spec.name
        run_name = (
            run_name_override
            if run_name_override is not None
            else f"{self.task.name}__{opt_spec.name}__{projector_name}__{projection}"
        )
        history: list[dict[str, Any]] = []

        with self._on_run_start(
            run_name=run_name,
            opt_spec=opt_spec,
            proj_spec=proj_spec,
            projection=projection,
            seed=seed,
            resume_from=resume_from,
        ):
            try:
                seed_everything(seed)

                model = self.task.model_factory().to(device=self.device)
                if self.config.dtype is not None:
                    model = model.to(dtype=self.config.dtype)
                model.train()

                params = list(model.parameters())

                if self.config.compile_model:
                    compile_kwargs: dict[str, Any] = {"dynamic": True}
                    if self.config.compile_mode is not None:
                        compile_kwargs["mode"] = self.config.compile_mode
                    model = torch.compile(model, **compile_kwargs)
                optimizer = opt_spec.build(params)
                projector = None if proj_spec is None else proj_spec.build(params)

                supports_projection = "projector" in inspect.signature(optimizer.step).parameters

                if resume_from is not None:
                    model.load_state_dict(resume_from.model_state)
                    optimizer.load_state_dict(resume_from.optimizer_state)
                    if (
                        resume_from.projector_state is not None
                        and projector is not None
                        and hasattr(projector, "load_state_dict")
                    ):
                        projector.load_state_dict(resume_from.projector_state)
                    torch.set_rng_state(resume_from.torch_rng_state)
                    if resume_from.cuda_rng_state is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state(resume_from.cuda_rng_state)
                    np.random.set_state(resume_from.numpy_rng_state)
                    random.setstate(resume_from.python_rng_state)

                train_iter = cycle(self.task.train_loader)
                basis_iter = cycle(self.task.basis_loader or self.task.train_loader)

                full_basis_batch: Batch | None = None
                if (
                    projector is not None
                    and proj_spec is not None
                    and proj_spec.basis_full_dataset
                ):
                    full_basis_batch = self._build_full_basis_batch(
                        subsample=proj_spec.basis_subsample,
                    )

                full_batch_chi_every = self.config.chi_k_full_batch_every
                if full_batch_chi_every is not None:
                    if full_batch_chi_every <= 0:
                        raise ValueError(
                            "chi_k_full_batch_every must be a positive integer or None."
                        )
                    if projector is None:
                        full_batch_chi_every = None
                    elif full_basis_batch is None:
                        raise ValueError(
                            "chi_k_full_batch_every requires a projector with "
                            "basis_full_dataset=True so a deterministic full-dataset "
                            "batch is cached."
                        )

                def get_basis_batch() -> Batch:
                    if full_basis_batch is not None:
                        return full_basis_batch
                    return self._next_basis_batch(basis_iter)

                if projector is not None and proj_spec is not None and proj_spec.update_before_train:
                    self._update_projector(
                        projector=projector,
                        proj_spec=proj_spec,
                        opt_spec=opt_spec,
                        optimizer=optimizer,
                        projection=projection,
                        model=model,
                        step=0,
                        basis_batch=get_basis_batch(),
                    )

                switching_enabled = (
                    resume_from is None
                    and proj_spec is not None
                )
                if resume_from is not None:
                    effective_projection: ProjectionMode = projection
                    switched_at_step: int | None = resume_from.step
                    chi_ema: float | None = resume_from.chi_ema
                else:
                    effective_projection = "none" if switching_enabled else projection
                    switched_at_step = None
                    chi_ema = None
                alpha = self.config.chi_ema_factor
                start_step = 1 if resume_from is None else resume_from.step + 1

                frozen_bulk = projection == "bulk" and self.config.frozen_bulk

                fgd_state = self._prepare_forward_gradient_state(model) if opt_spec.kind == "forward_gradient" else None
                start_time = time.perf_counter()

                last_log_time = start_time
                n_windows = 0

                loss_sum = 0.0
                loss_count = 0
                chi_sum = 0.0
                chi_count = 0
                chi_ema_sum = 0.0
                chi_ema_count = 0
                optimizer_finalized = False

                swa_from_step = self.config.swa_from_step
                swa_params: list[torch.Tensor] | None = None
                swa_count = 0

                swa_chi_ema: float | None = None

                for step in range(start_step, self.config.steps + 1):
                    batch = self._next_train_batch(train_iter)
                    should_log_step = (
                        step == 1
                        or step % self.config.log_every == 0
                        or step == self.config.steps
                    )

                    if (
                            projector is not None
                            and proj_spec is not None
                            and proj_spec.update_every_steps is not None
                            and step % proj_spec.update_every_steps == 0
                            and not (
                                frozen_bulk
                                and effective_projection != "none"
                                and getattr(projector, "is_ready", False)
                            )
                    ):
                        self._update_projector(
                            projector=projector,
                            proj_spec=proj_spec,
                            opt_spec=opt_spec,
                            optimizer=optimizer,
                            projection=projection,
                            model=model,
                            step=step,
                            basis_batch=get_basis_batch(),
                        )

                    if (
                            projector is not None
                            and proj_spec is not None
                            and effective_projection != "none"
                            and not getattr(projector, "is_ready", False)
                    ):
                        self._update_projector(
                            projector=projector,
                            proj_spec=proj_spec,
                            opt_spec=opt_spec,
                            optimizer=optimizer,
                            projection=projection,
                            model=model,
                            step=step,
                            basis_batch=get_basis_batch(),
                        )

                    loss_value, chi_k, subspace_usefulness_rho = self._optimizer_step(
                        model=model,
                        optimizer=optimizer,
                        opt_spec=opt_spec,
                        batch=batch,
                        projector=projector,
                        projection=effective_projection,
                        fgd_state=fgd_state,
                        supports_projection=supports_projection,
                        compute_chi_k=full_batch_chi_every is None,
                        compute_subspace_usefulness=should_log_step,
                    )

                    if swa_from_step is not None and step >= swa_from_step:
                        with torch.no_grad():
                            if swa_params is None:
                                swa_params = [p.detach().clone() for p in params]
                                swa_count = 1
                            else:
                                swa_count += 1
                                w = 1.0 / swa_count
                                for s, p in zip(swa_params, params):
                                    s.mul_(1.0 - w).add_(p.detach(), alpha=w)

                    if (
                        full_batch_chi_every is not None
                        and projector is not None
                        and getattr(projector, "is_ready", False)
                        and (step == 1 or step % full_batch_chi_every == 0)
                    ):
                        chi_k = self._compute_chi_k_full_batch(
                            model=model,
                            optimizer=optimizer,
                            opt_spec=opt_spec,
                            projector=projector,
                            basis_batch=full_basis_batch,
                            supports_projection=supports_projection,
                            fgd_state=fgd_state,
                        )

                    if chi_k is not None:
                        chi_ema = chi_k if chi_ema is None else alpha * chi_ema + (1.0 - alpha) * chi_k

                    loss_sum += loss_value
                    loss_count += 1
                    if chi_k is not None:
                        chi_sum += chi_k
                        chi_count += 1
                    if chi_ema is not None:
                        chi_ema_sum += chi_ema
                        chi_ema_count += 1

                    if switching_enabled and switched_at_step is None:
                        trigger_by_ema = (
                            proj_spec.switch_on_alignment_ema is not None
                            and chi_ema is not None
                            and chi_ema >= proj_spec.switch_on_alignment_ema
                        )
                        trigger_by_step = (
                            proj_spec.switch_on_step is not None
                            and step >= proj_spec.switch_on_step
                        )
                        trigger_by_sentinel = os.path.exists(FORCE_SWITCH_SENTINEL)
                        if trigger_by_ema or trigger_by_step or trigger_by_sentinel:
                            effective_projection = projection
                            switched_at_step = step

                            if (
                                frozen_bulk
                                and projector is not None
                                and proj_spec is not None
                            ):
                                self._update_projector(
                                    projector=projector,
                                    proj_spec=proj_spec,
                                    opt_spec=opt_spec,
                                    optimizer=optimizer,
                                    projection=projection,
                                    model=model,
                                    step=step,
                                    basis_batch=get_basis_batch(),
                                )
                            if on_switch is not None:
                                on_switch(self._capture_switch_checkpoint(
                                    step=step,
                                    model=model,
                                    optimizer=optimizer,
                                    projector=projector,
                                    chi_ema=chi_ema,
                                ))
                            if trigger_by_sentinel:
                                try:
                                    os.remove(FORCE_SWITCH_SENTINEL)
                                except OSError:
                                    pass

                    if step == self.config.steps and not optimizer_finalized:
                        optimizer_finalized = self._finalize_optimizer_for_eval(
                            optimizer=optimizer,
                            model=model,
                        )

                    if should_log_step:
                        now = time.perf_counter()
                        window_dt = now - last_log_time
                        n_windows += 1
                        window_avg = (now - start_time) / n_windows
                        last_log_time = now

                        loss_mean = loss_sum / loss_count
                        chi_mean = chi_sum / chi_count if chi_count > 0 else None
                        chi_ema_mean = chi_ema_sum / chi_ema_count if chi_ema_count > 0 else None
                        loss_sum = 0.0
                        loss_count = 0
                        chi_sum = 0.0
                        chi_count = 0
                        chi_ema_sum = 0.0
                        chi_ema_count = 0

                        if swa_params is not None:

                            measure_batch = (
                                full_basis_batch if full_basis_batch is not None else batch
                            )
                            refresh_basis = not (
                                frozen_bulk and switched_at_step is not None
                            )
                            row, chi_k_swa = self._swa_measure_row(
                                run_name=run_name,
                                opt_spec=opt_spec,
                                projector_name=projector_name,
                                projection=projection,
                                step=step,
                                model=model,
                                params=params,
                                swa_params=swa_params,
                                batch=batch,
                                measure_batch=measure_batch,
                                optimizer=optimizer,
                                proj_spec=proj_spec,
                                projector=projector,
                                effective_projection=effective_projection,
                                supports_projection=supports_projection,
                                fgd_state=fgd_state,
                                elapsed=now - start_time,
                                switched_at_step=switched_at_step,
                                epoch_time_sec=window_dt,
                                epoch_time_sec_avg=window_avg,
                                refresh_basis=refresh_basis,
                            )
                            if chi_k_swa is not None:
                                swa_chi_ema = (
                                    chi_k_swa
                                    if swa_chi_ema is None
                                    else alpha * swa_chi_ema + (1.0 - alpha) * chi_k_swa
                                )
                            row["chi_k_ema"] = swa_chi_ema
                            log_loss = row["loss"]
                        else:
                            stable_rank = None
                            probes = self.config.stable_rank_probes
                            if (
                                probes
                                and projector is not None
                                and getattr(projector, "is_ready", False)
                                and hasattr(projector, "estimate_stable_rank")
                            ):
                                sr_batch = (
                                    full_basis_batch
                                    if full_basis_batch is not None
                                    else get_basis_batch()
                                )
                                stable_rank = projector.estimate_stable_rank(
                                    self._loss_closure_for_basis(model, sr_batch),
                                    n_probes=probes,
                                )

                            row = self._make_log_row(
                                run_name=run_name,
                                opt_spec=opt_spec,
                                projector_name=projector_name,
                                projection=projection,
                                step=step,
                                loss_value=loss_mean,
                                model=model,
                                batch=batch,
                                optimizer=optimizer,
                                projector=projector,
                                elapsed=now - start_time,
                                chi_k=chi_mean,
                                chi_k_ema=chi_ema_mean,
                                effective_projection=effective_projection,
                                switched_at_step=switched_at_step,
                                epoch_time_sec=window_dt,
                                epoch_time_sec_avg=window_avg,
                                subspace_usefulness_rho=subspace_usefulness_rho,
                                stable_rank=stable_rank,
                            )
                            log_loss = loss_mean

                        history.append(row)
                        self._on_log_row(
                            row, step=step, loss_value=log_loss, model=model,
                        )

                        if self.config.show_progress:
                            self._print_progress(run_name, step, log_loss, row)

                if swa_params is not None:
                    with torch.no_grad():
                        for p, s in zip(params, swa_params):
                            p.copy_(s)

                self._on_run_finished(
                    run_name=run_name, history=history, model=model,
                )
                return RunResult(
                    run_name=run_name,
                    optimizer=opt_spec.name,
                    projector=projector_name,
                    projection=projection,
                    history=history,
                    model=model if self.config.keep_models else None,
                )

            except Exception:
                err = traceback.format_exc()
                self._on_run_failed(run_name, err)
                return RunResult(
                    run_name=run_name,
                    optimizer=opt_spec.name,
                    projector=projector_name,
                    projection=projection,
                    history=history,
                    status="failed",
                    error=err,
                )

    def _capture_switch_checkpoint(
            self,
            *,
            step: int,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            projector: Any | None,
            chi_ema: float | None,
    ) -> SwitchCheckpoint:
        model_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        optimizer_state = copy.deepcopy(optimizer.state_dict())

        def _to_cpu(obj: Any) -> Any:
            if torch.is_tensor(obj):
                return obj.detach().cpu().clone()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_cpu(v) for v in obj]
            return obj

        optimizer_state = _to_cpu(optimizer_state)

        projector_state = (
            _to_cpu(projector.state_dict())
            if projector is not None and hasattr(projector, "state_dict")
            else None
        )

        cuda_rng = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        return SwitchCheckpoint(
            step=step,
            model_state=model_state,
            optimizer_state=optimizer_state,
            projector_state=projector_state,
            chi_ema=float(chi_ema) if chi_ema is not None else None,
            torch_rng_state=torch.get_rng_state().clone(),
            cuda_rng_state=cuda_rng.clone() if cuda_rng is not None else None,
            numpy_rng_state=np.random.get_state(),
            python_rng_state=random.getstate(),
        )

    def _run_dom_then_bulk(
            self,
            entry: PlanEntry,
            seed: int,
    ) -> list[RunResult]:
        captured: dict[str, SwitchCheckpoint | None] = {"ckpt": None}

        def on_switch(ckpt: SwitchCheckpoint) -> None:
            captured["ckpt"] = ckpt

        dom_result = self._run_one(
            entry.opt_spec,
            entry.proj_spec,
            "dom",
            seed,
            on_switch=on_switch,
        )
        results = [dom_result]

        ckpt = captured["ckpt"]
        if dom_result.status != "ok":
            print(
                f"[{dom_result.run_name}] dom phase failed; skipping paired bulk run.",
                flush=True,
            )
            return results
        if ckpt is None:
            print(
                f"[{dom_result.run_name}] chi_ema never crossed "
                f"switch_on_alignment_ema during the dom phase; "
                f"skipping paired bulk run.",
                flush=True,
            )
            return results
        if ckpt.step >= self.config.steps:
            print(
                f"[{dom_result.run_name}] switch fired at step {ckpt.step} "
                f"(== config.steps); no remaining steps for bulk run, skipping.",
                flush=True,
            )
            return results

        projector_name = "none" if entry.proj_spec is None else entry.proj_spec.name
        bulk_run_name = (
            f"{self.task.name}__{entry.opt_spec.name}__{projector_name}__bulk"
        )
        bulk_result = self._run_one(
            entry.opt_spec,
            entry.proj_spec,
            "bulk",
            seed,
            resume_from=ckpt,
            run_name_override=bulk_run_name,
        )
        results.append(bulk_result)
        return results

    def _next_train_batch(self, train_iter: Iterable[Batch]) -> Batch:
        batch = next(train_iter)
        return self.task.batch_to_device(batch, self.device, self.config.dtype)

    def _next_basis_batch(self, basis_iter: Iterable[Batch]) -> Batch:
        batch = next(basis_iter)
        return self.task.batch_to_device(batch, self.device, self.config.dtype)

    def _loss_closure_for_basis(
            self,
            model: torch.nn.Module,
            basis_batch: Batch | None,
    ) -> Callable[[], torch.Tensor]:
        if basis_batch is None:
            raise RuntimeError("basis_batch is None; cannot build projector loss closure.")

        def closure() -> torch.Tensor:
            return self.task.loss_fn(model, basis_batch)

        return closure

    def _update_projector(
            self,
            projector: Any,
            proj_spec: ProjectorSpec,
            opt_spec: OptimizerSpec,
            optimizer: torch.optim.Optimizer | None,
            projection: ProjectionMode,
            model: torch.nn.Module,
            step: int,
            basis_batch: Batch | None,
    ) -> None:
        closure = self._loss_closure_for_basis(model, basis_batch)
        ctx = ProjectorContext(
            task=self.task,
            model=model,
            optimizer=optimizer,
            optimizer_spec=opt_spec,
            projector_spec=proj_spec,
            projection=projection,
            step=step,
            device=self.device,
            dtype=self.config.dtype,
            basis_batch=basis_batch,
            loss_closure=closure,
        )

        if proj_spec.update_kind == "custom":
            if proj_spec.update_fn is None:
                raise ValueError(f"ProjectorSpec({proj_spec.name!r}) has update_kind='custom' but update_fn=None.")
            proj_spec.update_fn(projector, ctx)
            return

        if proj_spec.update_kind == "loss_closure":
            projector.update_basis(closure)
            return

        if proj_spec.update_kind == "no_args":
            projector.update_basis()
            return

        raise ValueError(f"Unknown projector update kind: {proj_spec.update_kind}")

    def _optimizer_step(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            opt_spec: OptimizerSpec,
            batch: Batch,
            projector: Any | None,
            projection: ProjectionMode,
            fgd_state: dict[str, Any] | None,
            supports_projection: bool,
            compute_chi_k: bool = True,
            compute_subspace_usefulness: bool = False,
    ) -> tuple[float, float | None, float | None]:
        subspace_usefulness_rho: float | None = None

        if opt_spec.kind == "first_order":
            all_params = [p for group in optimizer.param_groups for p in group["params"]]
            train_params = [p for p in all_params if p.requires_grad]
            metric_requested = (
                compute_subspace_usefulness
                and projector is not None
                and projection != "none"
                and getattr(projector, "is_ready", False)
                and bool(train_params)
            )

            optimizer.zero_grad(set_to_none=True)
            loss = self.task.loss_fn(model, batch)
            loss_for_backward = loss if loss.ndim == 0 else loss.mean()
            loss_for_backward.backward()

            chi_params = (
                list(projector.params)
                if (
                    compute_chi_k
                    and projector is not None
                    and getattr(projector, "is_ready", False)
                )
                else []
            )
            prev_params = [p.detach().clone() for p in chi_params]

            if metric_requested:

                raw_update = self._raw_optimizer_update_for_metric(
                    optimizer,
                    projector=projector,
                    supports_projection=supports_projection,
                )
                projected_update = projector.project_update(raw_update, projection)
                raw_train: list[torch.Tensor] = []
                projected_train: list[torch.Tensor] = []
                for p, raw, projected in zip(all_params, raw_update, projected_update):
                    if not p.requires_grad:
                        continue
                    raw_train.append(raw.detach().to(device=p.device, dtype=p.dtype))
                    projected_train.append(projected.detach().to(device=p.device, dtype=p.dtype))

                metric_loss = self.task.loss_fn(model, batch)
                metric_loss = metric_loss if metric_loss.ndim == 0 else metric_loss.mean()
                metric_grads = torch.autograd.grad(
                    metric_loss,
                    train_params,
                    create_graph=True,
                    allow_unused=True,
                )
                subspace_usefulness_rho = self._compute_subspace_usefulness_rho(
                    params=train_params,
                    grads=metric_grads,
                    raw_update=raw_train,
                    projected_update=projected_train,
                )

            if supports_projection:
                optimizer.step(projector=projector, projection=projection)
            else:
                if projector is not None or projection != "none":
                    raise RuntimeError(
                        f"Optimizer {opt_spec.name!r} ({type(optimizer).__name__}) "
                        f"does not accept a 'projector' kwarg, but projection={projection!r} "
                        f"was requested. Use a projection-aware optimizer from src.optimizers."
                    )
                optimizer.step()

            chi_k_update: float | None = None
            if chi_params:
                if projection == "none":

                    for pv, p in zip(prev_params, chi_params):
                        pv.sub_(p.detach())
                    chi_k_update = projector.chi_k_of(_flatten(prev_params))
                else:

                    chi_k_update = _chi_k_from_alignment(optimizer, projection)
                    if chi_k_update is None:

                        for pv, p in zip(prev_params, chi_params):
                            pv.sub_(p.detach())
                        chi_k_update = projector.chi_k_of(_flatten(prev_params))

            return float(loss.detach().cpu()), chi_k_update, subspace_usefulness_rho

        if opt_spec.kind == "mezo":
            def closure() -> torch.Tensor:
                return self.task.loss_fn(model, batch)

            loss = optimizer.step(closure, projector=projector, projection=projection)
            value = _to_float(loss)
            if value is None:
                with torch.no_grad():
                    value = float(self.task.loss_fn(model, batch).detach().cpu())
            return value, None, None

        if opt_spec.kind == "forward_gradient":
            if fgd_state is None:
                raise RuntimeError("ForwardGradient state was not initialized.")

            names = fgd_state["names"]
            buffers = fgd_state["buffers"]
            base = fgd_state["base"]

            def closure(params: Sequence[torch.Tensor]) -> torch.Tensor:
                param_dict = {name: p for name, p in zip(names, params)}
                fmodel = _FunctionalModel(base, param_dict, buffers)
                return self.task.loss_fn(fmodel, batch)

            loss = optimizer.step(closure, projector=projector, projection=projection)
            value = _to_float(loss)
            if value is None:
                with torch.no_grad():
                    value = float(self.task.loss_fn(model, batch).detach().cpu())
            return value, None, None

        raise ValueError(f"Unknown optimizer kind: {opt_spec.kind}")

    def _raw_optimizer_update_for_metric(
            self,
            optimizer: torch.optim.Optimizer,
            *,
            projector: Any | None,
            supports_projection: bool,
    ) -> tuple[torch.Tensor, ...]:
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        param_backup = {p: p.detach().clone() for p in all_params}
        state_backup = {
            p: {
                k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in st.items()
            }
            for p, st in optimizer.state.items()
        }
        last_info_backup = getattr(optimizer, "last_info", None)

        if supports_projection:
            optimizer.step(projector=projector, projection="none")
        else:
            optimizer.step()

        update = tuple(param_backup[p] - p.detach() for p in all_params)

        with torch.no_grad():
            for p in all_params:
                p.copy_(param_backup[p])
        optimizer.state.clear()
        optimizer.state.update(state_backup)
        if last_info_backup is not None:
            optimizer.last_info = last_info_backup

        return update

    def _compute_subspace_usefulness_rho(
            self,
            *,
            params: Sequence[torch.Tensor],
            grads: Sequence[torch.Tensor | None],
            raw_update: Sequence[torch.Tensor],
            projected_update: Sequence[torch.Tensor],
    ) -> float | None:

        if (
            len(params) != len(grads)
            or len(params) != len(raw_update)
            or len(params) != len(projected_update)
        ):
            raise ValueError(
                "params, grads, raw_update and projected_update must have the same length."
            )

        if not params:
            return None

        try:
            raw_decrease = self._quadratic_predicted_decrease(
                grads,
                params,
                raw_update,
                retain_graph=True,
            )
            projected_decrease = self._quadratic_predicted_decrease(
                grads,
                params,
                projected_update,
                retain_graph=False,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                ("derivative for" in msg and "is not implemented" in msg)
                or "needed for gradient computation has been modified by an inplace operation" in msg
            ):
                return None
            raise

        raw_value = float(raw_decrease.detach().cpu())
        projected_value = float(projected_decrease.detach().cpu())
        if not np.isfinite(raw_value) or not np.isfinite(projected_value):
            return None

        scale = max(abs(raw_value), 1e-12)
        raw_score = torch.nn.functional.softplus(raw_decrease.detach() / scale)
        projected_score = torch.nn.functional.softplus(projected_decrease.detach() / scale)
        raw_score_value = float(raw_score.cpu())
        if raw_score_value == 0.0 or not np.isfinite(raw_score_value):
            return None
        return float((projected_score / raw_score).cpu())

    @staticmethod
    def _quadratic_predicted_decrease(
            grads: Sequence[torch.Tensor | None],
            params: Sequence[torch.Tensor],
            vector: Sequence[torch.Tensor],
            *,
            retain_graph: bool,
    ) -> torch.Tensor:
        dot_terms = [
            (g * v).sum()
            for g, v in zip(grads, vector)
            if g is not None
        ]
        if not dot_terms:
            return vector[0].new_zeros(())

        first_order = dot_terms[0]
        for term in dot_terms[1:]:
            first_order = first_order + term

        if first_order.requires_grad:
            hvps = torch.autograd.grad(
                first_order,
                params,
                retain_graph=retain_graph,
                allow_unused=True,
            )
        else:
            hvps = tuple(None for _ in params)

        quadratic_terms = [
            (v * hv).sum()
            for v, hv in zip(vector, hvps)
            if hv is not None
        ]
        quadratic = vector[0].new_zeros(())
        for term in quadratic_terms:
            quadratic = quadratic + term

        return first_order - 0.5 * quadratic

    def _compute_chi_k_full_batch(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            opt_spec: OptimizerSpec,
            projector: Any,
            basis_batch: Batch,
            supports_projection: bool,
            fgd_state: dict[str, Any] | None = None,
    ) -> float:
        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        if opt_spec.kind == "first_order":
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = self.task.loss_fn(model, basis_batch)
                if loss.ndim != 0:
                    loss = loss.mean()
                loss.backward()

        param_backup = {p: p.detach().clone() for p in all_params}
        state_backup = {
            p: {
                k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in st.items()
            }
            for p, st in optimizer.state.items()
        }
        last_info_backup = getattr(optimizer, "last_info", None)
        gen = getattr(optimizer, "_generator", None)
        gen_backup = gen.get_state() if gen is not None else None

        self._virtual_raw_step(
            optimizer=optimizer,
            opt_spec=opt_spec,
            model=model,
            batch=basis_batch,
            projector=projector,
            supports_projection=supports_projection,
            fgd_state=fgd_state,
        )

        update = [param_backup[p] - p.detach() for p in projector.params]
        chi_k = projector.chi_k_of(_flatten(update))

        with torch.no_grad():
            for p in all_params:
                p.copy_(param_backup[p])
        optimizer.state.clear()
        optimizer.state.update(state_backup)
        if last_info_backup is not None:
            optimizer.last_info = last_info_backup
        if gen_backup is not None:
            gen.set_state(gen_backup)
        optimizer.zero_grad(set_to_none=True)

        return chi_k

    def _virtual_raw_step(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            opt_spec: OptimizerSpec,
            model: torch.nn.Module,
            batch: Batch,
            projector: Any,
            supports_projection: bool,
            fgd_state: dict[str, Any] | None,
    ) -> None:
        if opt_spec.kind == "first_order":
            if supports_projection:
                optimizer.step(projector=projector, projection="none")
            else:
                optimizer.step()
            return

        if opt_spec.kind == "mezo":
            def closure() -> torch.Tensor:
                loss = self.task.loss_fn(model, batch)
                return loss if loss.ndim == 0 else loss.mean()

            optimizer.step(closure, projector=projector, projection="none")
            return

        if opt_spec.kind == "forward_gradient":
            if fgd_state is None:
                raise RuntimeError("ForwardGradient state was not initialized.")
            names = fgd_state["names"]
            buffers = fgd_state["buffers"]
            base = fgd_state["base"]

            def closure(params: Sequence[torch.Tensor]) -> torch.Tensor:
                param_dict = {name: p for name, p in zip(names, params)}
                fmodel = _FunctionalModel(base, param_dict, buffers)
                loss = self.task.loss_fn(fmodel, batch)
                return loss if loss.ndim == 0 else loss.mean()

            optimizer.step(closure, projector=projector, projection="none")
            return

        raise ValueError(f"Unknown optimizer kind: {opt_spec.kind}")

    def _build_full_basis_batch(self, subsample: int | None = None) -> Batch:
        loader = self.task.train_loader
        accumulated: list[Batch] = []
        total = 0
        for batch in loader:
            accumulated.append(batch)
            total += _batch_size(batch)
            if subsample is not None and total >= subsample:
                break

        if not accumulated:
            raise RuntimeError("Cannot build full basis batch: empty train_loader.")

        first = accumulated[0]
        if torch.is_tensor(first):
            big = torch.cat(accumulated, dim=0)
        elif isinstance(first, (tuple, list)):
            n_fields = len(first)
            big_fields = [
                torch.cat([b[i] for b in accumulated], dim=0) for i in range(n_fields)
            ]
            big = type(first)(big_fields)
        elif isinstance(first, Mapping):
            big = type(first)(
                {k: torch.cat([b[k] for b in accumulated], dim=0) for k in first}
            )
        else:
            raise TypeError(
                f"Unsupported batch type for full-dataset Hessian: {type(first).__name__}."
            )

        if subsample is not None:
            big = _slice_batch(big, subsample)

        return self.task.batch_to_device(big, self.device, self.config.dtype)

    def _prepare_forward_gradient_state(self, model: torch.nn.Module) -> dict[str, Any]:
        names = list(dict(model.named_parameters()).keys())
        buffers = dict(model.named_buffers())
        base = copy.deepcopy(model).to("meta")
        return {"names": names, "buffers": buffers, "base": base}

    def _swa_measure_row(
            self,
            *,
            run_name: str,
            opt_spec: OptimizerSpec,
            projector_name: str,
            projection: ProjectionMode,
            step: int,
            model: torch.nn.Module,
            params: Sequence[torch.nn.Parameter],
            swa_params: Sequence[torch.Tensor],
            batch: Batch,
            measure_batch: Batch,
            optimizer: torch.optim.Optimizer,
            proj_spec: ProjectorSpec | None,
            projector: Any | None,
            effective_projection: ProjectionMode,
            supports_projection: bool,
            fgd_state: dict[str, Any] | None,
            elapsed: float,
            switched_at_step: int | None,
            epoch_time_sec: float | None,
            epoch_time_sec_avg: float | None,
            refresh_basis: bool,
    ) -> tuple[dict[str, Any], float | None]:
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        train_params = [p for p in all_params if p.requires_grad]

        param_backup = [p.detach().clone() for p in params]
        last_info_backup = getattr(optimizer, "last_info", None)
        basis_backup = projector.basis if projector is not None else None
        eigvals_backup = projector.eigvals if projector is not None else None
        gen = getattr(optimizer, "_generator", None)
        gen_backup = gen.get_state() if gen is not None else None
        was_training = model.training

        chi_k_swa: float | None = None
        stable_rank: float | None = None
        rho: float | None = None
        info = None

        try:
            with torch.no_grad():
                for p, w in zip(params, swa_params):
                    p.copy_(w)

            if refresh_basis and projector is not None and proj_spec is not None:
                self._update_projector(
                    projector=projector,
                    proj_spec=proj_spec,
                    opt_spec=opt_spec,
                    optimizer=optimizer,
                    projection=projection,
                    model=model,
                    step=step,
                    basis_batch=measure_batch,
                )

            projector_ready = projector is not None and getattr(projector, "is_ready", False)

            if projector_ready:
                chi_k_swa = self._compute_chi_k_full_batch(
                    model=model,
                    optimizer=optimizer,
                    opt_spec=opt_spec,
                    projector=projector,
                    basis_batch=measure_batch,
                    supports_projection=supports_projection,
                    fgd_state=fgd_state,
                )

            if opt_spec.kind == "first_order" and projector_ready and train_params:
                optimizer.zero_grad(set_to_none=True)
                loss_b = self.task.loss_fn(model, batch)
                (loss_b if loss_b.ndim == 0 else loss_b.mean()).backward()

                raw_update = self._raw_optimizer_update_for_metric(
                    optimizer,
                    projector=projector,
                    supports_projection=supports_projection,
                )
                projected_update = projector.project_update(raw_update, effective_projection)
                info = projector.info_for(raw_update, projected_update)

                if effective_projection != "none":
                    raw_train: list[torch.Tensor] = []
                    projected_train: list[torch.Tensor] = []
                    for p, raw, projected in zip(all_params, raw_update, projected_update):
                        if not p.requires_grad:
                            continue
                        raw_train.append(raw.detach().to(device=p.device, dtype=p.dtype))
                        projected_train.append(projected.detach().to(device=p.device, dtype=p.dtype))
                    metric_loss = self.task.loss_fn(model, batch)
                    metric_loss = metric_loss if metric_loss.ndim == 0 else metric_loss.mean()
                    metric_grads = torch.autograd.grad(
                        metric_loss, train_params, create_graph=True, allow_unused=True,
                    )
                    rho = self._compute_subspace_usefulness_rho(
                        params=train_params,
                        grads=metric_grads,
                        raw_update=raw_train,
                        projected_update=projected_train,
                    )

            with torch.no_grad():
                loss_t = self.task.loss_fn(model, measure_batch)
                swa_loss = float((loss_t if loss_t.ndim == 0 else loss_t.mean()).detach().cpu())

            probes = self.config.stable_rank_probes
            if probes and projector_ready and hasattr(projector, "estimate_stable_rank"):
                stable_rank = projector.estimate_stable_rank(
                    self._loss_closure_for_basis(model, measure_batch),
                    n_probes=probes,
                )

            row = self._make_log_row(
                run_name=run_name,
                opt_spec=opt_spec,
                projector_name=projector_name,
                projection=projection,
                step=step,
                loss_value=swa_loss,
                model=model,
                batch=batch,
                optimizer=optimizer,
                projector=projector,
                elapsed=elapsed,
                chi_k=chi_k_swa,
                chi_k_ema=None,
                effective_projection=effective_projection,
                switched_at_step=switched_at_step,
                epoch_time_sec=epoch_time_sec,
                epoch_time_sec_avg=epoch_time_sec_avg,
                subspace_usefulness_rho=rho,
                stable_rank=stable_rank,
            )

            if info is not None:
                row["update/raw_update_norm"] = info.raw_norm
                row["update/projected_update_norm"] = info.projected_norm
                row["update/alignment"] = info.alignment
        finally:
            if last_info_backup is not None:
                optimizer.last_info = last_info_backup
            if projector is not None:
                projector.basis = basis_backup
                projector.eigvals = eigvals_backup
            if gen_backup is not None:
                gen.set_state(gen_backup)
            with torch.no_grad():
                for p, b in zip(params, param_backup):
                    p.copy_(b)
            optimizer.zero_grad(set_to_none=True)
            if was_training and not model.training:
                model.train()

        return row, chi_k_swa

    def _finalize_optimizer_for_eval(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            model: torch.nn.Module,
    ) -> bool:
        apply_weights = getattr(optimizer, "apply_swa_weights", None)
        if not callable(apply_weights):
            return False

        applied = bool(apply_weights())
        if applied and getattr(optimizer, "update_bn_on_finalize", False):
            self._update_batch_norm_for_model(model)
        return applied

    def _update_batch_norm_for_model(self, model: torch.nn.Module) -> None:
        bn_modules = [
            module for module in model.modules() if isinstance(module, _BatchNorm)
        ]
        if not bn_modules:
            return

        momenta = {module: module.momentum for module in bn_modules}
        was_training = model.training

        for module in bn_modules:
            module.reset_running_stats()

        n_seen = 0
        model.train()
        try:
            with torch.no_grad():
                for batch in self.task.train_loader:
                    batch_size = _batch_size(batch)
                    if batch_size <= 0:
                        continue
                    momentum = batch_size / float(n_seen + batch_size)
                    for module in bn_modules:
                        module.momentum = momentum
                    n_seen += batch_size

                    if self.task.batch_to_device is not None:
                        batch = self.task.batch_to_device(
                            batch,
                            self.device,
                            self.config.dtype,
                        )
                    loss = self.task.loss_fn(model, batch)
                    if torch.is_tensor(loss):
                        loss.detach()
        finally:
            for module, momentum in momenta.items():
                module.momentum = momentum
            model.train(was_training)

    def _on_run_start(
            self,
            *,
            run_name: str,
            opt_spec: OptimizerSpec,
            proj_spec: ProjectorSpec | None,
            projection: ProjectionMode,
            seed: int,
            resume_from: SwitchCheckpoint | None = None,
    ) -> AbstractContextManager[None]:
        del run_name, opt_spec, proj_spec, projection, seed, resume_from
        return nullcontext()

    def _on_log_row(
            self,
            row: dict[str, Any],
            *,
            step: int,
            loss_value: float,
            model: torch.nn.Module,
    ) -> None:
        del step, loss_value, model
        self._append_jsonl(row["run"], row)

    def _on_run_finished(
            self,
            *,
            run_name: str,
            history: list[dict[str, Any]],
            model: torch.nn.Module,
    ) -> None:
        del run_name, history, model

    def _on_run_failed(self, run_name: str, error: str) -> None:
        del run_name, error

    def _make_log_row(
            self,
            run_name: str,
            opt_spec: OptimizerSpec,
            projector_name: str,
            projection: ProjectionMode,
            step: int,
            loss_value: float,
            model: torch.nn.Module,
            batch: Batch,
            optimizer: torch.optim.Optimizer,
            projector: Any | None,
            elapsed: float,
            *,
            chi_k: float | None = None,
            chi_k_ema: float | None = None,
            effective_projection: ProjectionMode | None = None,
            switched_at_step: int | None = None,
            epoch_time_sec: float | None = None,
            epoch_time_sec_avg: float | None = None,
            subspace_usefulness_rho: float | None = None,
            stable_rank: float | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run": run_name,
            "task": self.task.name,
            "optimizer": opt_spec.name,
            "projector": projector_name,
            "projection": projection,
            "effective_projection": effective_projection if effective_projection is not None else projection,
            "switched_at_step": switched_at_step,
            "step": step,
            "loss": loss_value,
            "elapsed_sec": elapsed,
            "chi_k": chi_k,
            "chi_k_ema": chi_k_ema,
            "epoch_time_sec": epoch_time_sec,
            "epoch_time_sec_avg": epoch_time_sec_avg,
            "subspace_usefulness/rho": subspace_usefulness_rho
        }
        if stable_rank is not None:
            row["stable_rank"] = stable_rank

        if self.task.metrics_fn is not None:
            model_was_training = model.training
            model.eval()
            with torch.no_grad():
                metrics = self.task.metrics_fn(model, batch)
            if model_was_training:
                model.train()
            row.update({k: _jsonable(v) for k, v in metrics.items()})

        last_info = getattr(optimizer, "last_info", None)
        if isinstance(last_info, Mapping):
            for key, value in last_info.items():
                row[f"update/{key}"] = _jsonable(value)

        subspace_pct = self._projector_subspace_pct(
            projector,
            effective_projection if effective_projection is not None else projection,
        )
        if subspace_pct is not None:
            row["projector/subspace_pct"] = subspace_pct

        n = self.config.log_top_eigvals
        if n and projector is not None:
            eigvals = getattr(projector, "eigvals", None)
            if torch.is_tensor(eigvals) and eigvals.numel() > 0:
                top = torch.sort(
                    eigvals.detach().float().flatten(), descending=True
                ).values[:n]
                for i, v in enumerate(top.tolist()):
                    row[f"eigval/{i}"] = v

        diag_fn = getattr(projector, "precond_diagnostics", None)
        if callable(diag_fn):
            diag = diag_fn()
            if isinstance(diag, Mapping):
                for key, value in diag.items():
                    row[f"precond/{key}"] = _jsonable(value)

        return row

    @staticmethod
    def _projector_subspace_pct(
            projector: Any | None,
            projection: ProjectionMode,
    ) -> float | None:
        if projector is None or not getattr(projector, "is_ready", False):
            return None
        if projection == "none":
            return None

        n_params = getattr(projector, "n_params", None)
        if n_params is None or int(n_params) <= 0:
            return None

        dim = getattr(projector, "projection_dim", None)
        if dim is not None:
            value = dim() if callable(dim) else dim
            if value is None:
                return None
            active_dim = float(value)
            if projection == "bulk":
                active_dim = float(n_params) - active_dim
            return 100.0 * active_dim / float(n_params)

        basis = getattr(projector, "basis", None)
        if torch.is_tensor(basis) and basis.ndim == 2:
            active_dim = float(basis.shape[1])
            if projection == "bulk":
                active_dim = float(n_params) - active_dim
            return 100.0 * active_dim / float(n_params)

        return None

    def _print_progress(
            self,
            run_name: str,
            step: int,
            loss_value: float,
            row: Mapping[str, Any],
    ) -> None:
        msg = f"[{run_name}] step={step}/{self.config.steps} loss={loss_value:.6g}"
        if "accuracy" in row:
            msg += f" acc={row['accuracy']:.4f}"
        print(msg)

    def _append_jsonl(self, run_name: str, row: dict[str, Any]) -> None:
        if self.save_dir is None:
            return
        path = self.save_dir / f"{run_name}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def _save_result(self, result: RunResult) -> None:
        if self.save_dir is None:
            return
        path = self.save_dir / f"{result.run_name}.result.json"
        payload = {
            "run_name": result.run_name,
            "optimizer": result.optimizer,
            "projector": result.projector,
            "projection": result.projection,
            "status": result.status,
            "error": result.error,
            "final_loss": result.final_loss,
            "history": result.history
        }
        path.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_summary(self, results: Sequence[RunResult]) -> None:
        if self.save_dir is None:
            return
        summary = [
            {
                "run_name": r.run_name,
                "optimizer": r.optimizer,
                "projector": r.projector,
                "projection": r.projection,
                "status": r.status,
                "final_loss": r.final_loss,
                "error": r.error
            }
            for r in results
        ]
        (self.save_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def results_to_rows(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result.history)
    return rows


def results_to_dataframe(results: Sequence[RunResult]):
    import pandas as pd

    return pd.DataFrame(results_to_rows(results))
