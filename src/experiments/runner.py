"""
Сюда вынесена только инфраструктура эксперимента без конкретики по моделям и датасетам

ExperimentRunner получает уже готовый TaskSpec и запускает комбинации:
optimizer + projector + projection_mode
"""

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
    """
    Для логов, приводит к json-совместимым объектам
    """
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
    """
    Интерфейс для задачи (Task), используемый раннером экспериментов
    """

    name: str
    model_factory: Callable[[], torch.nn.Module]
    train_loader: Iterable[Batch]
    loss_fn: LossFn
    metrics_fn: MetricsFn | None = None
    batch_to_device: BatchToDeviceFn = None
    basis_loader: Iterable[Batch] | None = None


@dataclass
class OptimizerSpec:
    """
    Описывает и создаёт один оптимизатор
    name - имя для логов
    cls - класс (Adam, SGD, ...)
    kwargs - параметры оптимизатора
    kind - тип оптимизатора
    factory - функция-фабрика для класса оптимизатора

    Можно создать через cls или через factory
    Второе, если создаём как-то нестандартно
    """

    name: str
    cls: type | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: OptimizerKind = "first_order"
    factory: Callable[
        [Sequence[torch.nn.Parameter]], torch.optim.Optimizer
    ] | None = None

    def build(self, params: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if self.factory is not None:
            return self.factory(params)
        if self.cls is None:
            raise ValueError(
                f"OptimizerSpec({self.name!r}) has neither cls nor factory."
            )
        return self.cls(params, **self.kwargs)


@dataclass
class ProjectorSpec:
    """
    Описывает проектор

    name - имя для логов
    cls - класс проектора
    kwargs - параметры конструктора
    modes - какие режимы запускать: dom/bulk/none
    update_kind - как обновлять basis
    update_before_train - обновить ли basis до первого шага
    update_every_steps - как часто обновлять basis во время обучения
    update_fn - кастомное обновление basis
    """

    name: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    modes: Sequence[ProjectionMode] = ("dom", "bulk")
    update_kind: ProjectorUpdateKind = "no_args"
    update_before_train: bool = True
    update_every_steps: int | None = None
    update_fn: ProjectorUpdateFn | None = None

    basis_full_dataset: bool = False
    basis_subsample: int | None = None
    switch_on_alignment_ema: float | None = None
    switch_on_step: int | None = None

    def build(self, params: Sequence[torch.nn.Parameter]) -> Any:
        return self.cls(params, **self.kwargs)


@dataclass
class RunnerConfig:
    """
    Глобальные настройки запуска

    steps - число train steps
    device - cuda/cpu
    dtype - dtype модели и float-тензоров batch
    seed - базовый seed
    log_every - как часто писать логи
    include_baseline - запускать ли optimizer без проекции
    fail_fast - падать ли при первой ошибке
    save_dir - куда сохранять json/jsonl логи
    keep_models - возвращать ли обученные модели в RunResult
    show_progress - печатать ли progress
    """

    steps: int
    device: str | torch.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
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

    chi_ema_factor: float = 0.9
    chi_k_full_batch_every: int | None = None
    compile_model: bool = False
    compile_mode: str | None = None


@dataclass
class ProjectorContext:
    """
    Контекст для кастомного обновления проектора

    Используется только если proj_spec.update_kind == "custom"
    В таком случае вызывается proj_spec.update_fn(projector, ctx)
    """

    task: TaskSpec
    model: torch.nn.Module
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
    """
    Результат одного запуска
    Один запуск - одна комбинация:
    task + optimizer + projector + projection_mode
    """

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
    """
    Обертка для ForwardGradient
    Делает объект который можно вызывать как модель
    """

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
    """
    Запускает оптимизатор + проектор для одного TaskSpec
    """

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

        for run_idx, entry in enumerate(plan):
            seed = self.config.seed + run_idx
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
        """
        Строит список PlanEntry:
        task + optimizer + projector + projection_mode
        """
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
                plan.append(
                    PlanEntry(opt_spec=opt_spec, proj_spec=proj_spec, modes=modes)
                )

        for opt_spec in self.optimizers:
            if self.config.include_baseline:
                add(opt_spec, None, ("none",))

            for proj_spec in self.projectors:
                subspace_modes = tuple(m for m in proj_spec.modes if m != "none")
                has_none = any(m == "none" for m in proj_spec.modes)
                if has_none:
                    add(opt_spec, None, ("none",))

                if subspace_modes == ("dom", "bulk") or subspace_modes == (
                    "bulk",
                    "dom",
                ):
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

                supports_projection = (
                    "projector" in inspect.signature(optimizer.step).parameters
                )

                if resume_from is not None:
                    model.load_state_dict(resume_from.model_state)
                    optimizer.load_state_dict(resume_from.optimizer_state)
                    torch.set_rng_state(resume_from.torch_rng_state)
                    if (
                        resume_from.cuda_rng_state is not None
                        and torch.cuda.is_available()
                    ):
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

                if (
                    projector is not None
                    and proj_spec is not None
                    and proj_spec.update_before_train
                ):
                    self._update_projector(
                        projector=projector,
                        proj_spec=proj_spec,
                        opt_spec=opt_spec,
                        projection=projection,
                        model=model,
                        step=0,
                        basis_batch=get_basis_batch(),
                    )

                switching_enabled = resume_from is None and proj_spec is not None
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

                fgd_state = (
                    self._prepare_forward_gradient_state(model)
                    if opt_spec.kind == "forward_gradient"
                    else None
                )
                start_time = time.perf_counter()
                last_log_time = start_time
                n_windows = 0
                loss_sum = 0.0
                loss_count = 0
                chi_sum = 0.0
                chi_count = 0
                chi_ema_sum = 0.0
                chi_ema_count = 0

                for step in range(start_step, self.config.steps + 1):
                    batch = self._next_train_batch(train_iter)

                    if (
                        projector is not None
                        and proj_spec is not None
                        and proj_spec.update_every_steps is not None
                        and step % proj_spec.update_every_steps == 0
                    ):
                        self._update_projector(
                            projector=projector,
                            proj_spec=proj_spec,
                            opt_spec=opt_spec,
                            projection=projection,
                            model=model,
                            step=step,
                            basis_batch=get_basis_batch(),
                        )

                    loss_value, chi_k_grad = self._optimizer_step(
                        model=model,
                        optimizer=optimizer,
                        opt_spec=opt_spec,
                        batch=batch,
                        projector=projector,
                        projection=effective_projection,
                        fgd_state=fgd_state,
                        supports_projection=supports_projection,
                        compute_chi_k=full_batch_chi_every is None,
                    )

                    if (
                        full_batch_chi_every is not None
                        and projector is not None
                        and getattr(projector, "is_ready", False)
                        and (step == 1 or step % full_batch_chi_every == 0)
                    ):
                        chi_k_grad = self._compute_chi_k_full_batch(
                            model=model,
                            projector=projector,
                            basis_batch=full_basis_batch,
                        )

                    if chi_k_grad is not None:
                        chi_ema = (
                            chi_k_grad
                            if chi_ema is None
                            else alpha * chi_ema + (1.0 - alpha) * chi_k_grad
                        )

                    loss_sum += loss_value
                    loss_count += 1
                    if chi_k_grad is not None:
                        chi_sum += chi_k_grad
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
                            if on_switch is not None:
                                on_switch(
                                    self._capture_switch_checkpoint(
                                        step=step,
                                        model=model,
                                        optimizer=optimizer,
                                        chi_ema=chi_ema,
                                    )
                                )
                            if trigger_by_sentinel:
                                try:
                                    os.remove(FORCE_SWITCH_SENTINEL)
                                except OSError:
                                    pass

                    if (
                        step == 1
                        or step % self.config.log_every == 0
                        or step == self.config.steps
                    ):
                        now = time.perf_counter()
                        window_dt = now - last_log_time
                        n_windows += 1
                        window_avg = (now - start_time) / n_windows
                        last_log_time = now

                        loss_mean = loss_sum / loss_count
                        chi_mean = chi_sum / chi_count if chi_count > 0 else None
                        chi_ema_mean = (
                            chi_ema_sum / chi_ema_count if chi_ema_count > 0 else None
                        )
                        loss_sum = 0.0
                        loss_count = 0
                        chi_sum = 0.0
                        chi_count = 0
                        chi_ema_sum = 0.0
                        chi_ema_count = 0

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
                            elapsed=now - start_time,
                            chi_k_grad=chi_mean,
                            chi_k_grad_ema=chi_ema_mean,
                            effective_projection=effective_projection,
                            switched_at_step=switched_at_step,
                            epoch_time_sec=window_dt,
                            epoch_time_sec_avg=window_avg,
                        )
                        history.append(row)
                        self._on_log_row(
                            row,
                            step=step,
                            loss_value=loss_mean,
                            model=model,
                        )

                        if self.config.show_progress:
                            self._print_progress(run_name, step, loss_mean, row)

                self._on_run_finished(
                    run_name=run_name,
                    history=history,
                    model=model,
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

        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        return SwitchCheckpoint(
            step=step,
            model_state=model_state,
            optimizer_state=optimizer_state,
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
            seed + 1,
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
            raise RuntimeError(
                "basis_batch is None; cannot build projector loss closure."
            )

        def closure() -> torch.Tensor:
            return self.task.loss_fn(model, basis_batch)

        return closure

    def _update_projector(
        self,
        projector: Any,
        proj_spec: ProjectorSpec,
        opt_spec: OptimizerSpec,
        projection: ProjectionMode,
        model: torch.nn.Module,
        step: int,
        basis_batch: Batch | None,
    ) -> None:
        closure = self._loss_closure_for_basis(model, basis_batch)
        ctx = ProjectorContext(
            task=self.task,
            model=model,
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
                raise ValueError(
                    f"ProjectorSpec({proj_spec.name!r}) has update_kind='custom' but update_fn=None."
                )
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
    ) -> tuple[float, float | None]:
        if opt_spec.kind == "first_order":
            optimizer.zero_grad(set_to_none=True)
            loss = self.task.loss_fn(model, batch)
            loss.backward()

            chi_k_grad: float | None = None
            if (
                compute_chi_k
                and projector is not None
                and getattr(projector, "is_ready", False)
            ):
                trainable_grads = [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).detach()
                    for p in projector.params
                ]
                if trainable_grads:
                    flat_grad = _flatten(trainable_grads)
                    chi_k_grad = projector.chi_k_of(flat_grad)

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
            return float(loss.detach().cpu()), chi_k_grad

        if opt_spec.kind == "mezo":

            def closure() -> torch.Tensor:
                return self.task.loss_fn(model, batch)

            loss = optimizer.step(closure, projector=projector, projection=projection)
            value = _to_float(loss)
            if value is None:
                with torch.no_grad():
                    value = float(self.task.loss_fn(model, batch).detach().cpu())
            return value, None

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
            return value, None

        raise ValueError(f"Unknown optimizer kind: {opt_spec.kind}")

    def _compute_chi_k_full_batch(
        self,
        model: torch.nn.Module,
        projector: Any,
        basis_batch: Batch,
    ) -> float:
        params = list(projector.params)
        for p in params:
            if p.grad is not None:
                p.grad = None

        with torch.enable_grad():
            loss = self.task.loss_fn(model, basis_batch)
            if loss.ndim != 0:
                loss = loss.mean()
            loss.backward()

        grads = [
            (p.grad if p.grad is not None else torch.zeros_like(p)).detach()
            for p in params
        ]
        flat_grad = _flatten(grads)
        chi_k = projector.chi_k_of(flat_grad)

        for p in params:
            if p.grad is not None:
                p.grad = None

        return chi_k

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
        elapsed: float,
        *,
        chi_k_grad: float | None = None,
        chi_k_grad_ema: float | None = None,
        effective_projection: ProjectionMode | None = None,
        switched_at_step: int | None = None,
        epoch_time_sec: float | None = None,
        epoch_time_sec_avg: float | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run": run_name,
            "task": self.task.name,
            "optimizer": opt_spec.name,
            "projector": projector_name,
            "projection": projection,
            "effective_projection": effective_projection
            if effective_projection is not None
            else projection,
            "switched_at_step": switched_at_step,
            "step": step,
            "loss": loss_value,
            "elapsed_sec": elapsed,
            "chi_k_grad": chi_k_grad,
            "chi_k_grad_ema": chi_k_grad_ema,
            "epoch_time_sec": epoch_time_sec,
            "epoch_time_sec_avg": epoch_time_sec_avg,
        }

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

        return row

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
            "history": result.history,
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
                "error": r.error,
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
