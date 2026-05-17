"""
Конфигурации экспериментов для репликации работы Song et al. (ICLR 2025)

Настройки из статьи:
MLP/MNIST-5k, CNN/CIFAR10-5k и Transformer/SST2-1k
обучаются с параметрами из Appendix B: соответствующие lr, batch size, MSE, k и 20000 шагов
"""

from __future__ import annotations

from typing import Callable

import torch

from src.experiments.runner import OptimizerSpec, ProjectorSpec, RunnerConfig, TaskSpec
from src.experiments.tasks import (
    make_cifar_cnn3_task,
    make_mnist_mlp3_task,
    make_sst2_transformer_task,
)
from src.optimizers import SGD, SGDM, Adam, ForwardGradient, MeZO, Muon
from src.projections import HessianEigenspaceProjector

PAPER_TASKS: dict[str, dict] = {
    "mnist_mlp3": {
        "task_factory": lambda: make_mnist_mlp3_task(
            batch_size=50,
            loss_type="mse",
            num_classes=10,
        ),
        "k": 10,
        "lr": 0.01,
        "steps": 20000,
    },
    "cifar10_cnn3": {
        "task_factory": lambda: make_cifar_cnn3_task(
            batch_size=50,
            loss_type="mse",
            num_classes=10,
        ),
        "k": 10,
        "lr": 0.001,
        "steps": 20000,
    },
    "sst2_transformer": {
        "task_factory": lambda: make_sst2_transformer_task(
            batch_size=50,
            loss_type="mse",
            num_classes=2,
        ),
        "k": 2,
        "lr": 0.001,
        "steps": 20000,
    },
}

def paper_projector_spec(
    k: int,
    seed: int = 0,
    solver: str = "eigsh",
    *,
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    maxiter: int | None = None,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> ProjectorSpec:
    """
    Проектор на top-k направлений Гессиана по Song et al. (Секция 3.2)

    Базис пересчитывается каждый шаг по всему train set, а переключение на режим с проекцией
    происходит при EMA(chi_k) > switch_on_alignment_ema
    """
    kwargs: dict = {"k": k, "solver": solver, "tol": 1e-4, "seed": seed}
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    return ProjectorSpec(
        name="hessian_topk",
        cls=HessianEigenspaceProjector,
        kwargs=kwargs,
        modes=("dom", "bulk"),
        update_kind="loss_closure",
        update_before_train=True,
        update_every_steps=update_every_steps,
        basis_full_dataset=True,
        basis_subsample=basis_subsample,
        switch_on_alignment_ema=switch_on_alignment_ema,
        switch_on_step=switch_on_step,
    )

def paper_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    return [
        OptimizerSpec(name="sgd", cls=SGD, kwargs={"lr": lr}, kind="first_order"),
    ]


def extended_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """
    SGD + Adam + Muon + MeZO + ForwardGradient
    """
    return [
        OptimizerSpec(name="sgd", cls=SGD, kwargs={"lr": lr}, kind="first_order"),
        OptimizerSpec(
            name="adam",
            cls=Adam,
            kwargs={"lr": min(lr, 1e-3)},
            kind="first_order",
        ),
        OptimizerSpec(
            name="muon",
            cls=Muon,
            kwargs={"lr": min(lr * 2, 0.02), "momentum": 0.95, "nesterov": True},
            kind="first_order",
        ),
        OptimizerSpec(
            name="mezo",
            cls=MeZO,
            kwargs={"lr": 1e-4, "eps": 1e-3, "seed": 0},
            kind="mezo",
        ),
        OptimizerSpec(
            name="forward_gradient",
            cls=ForwardGradient,
            kwargs={"lr": 1e-4, "seed": 0},
            kind="forward_gradient",
        ),
    ]

def transformer_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """
    Adam + Muon
    """
    return [
        OptimizerSpec(
            name="adam",
            cls=Adam,
            kwargs={"lr": min(lr, 1e-3)},
            kind="first_order",
        ),
        OptimizerSpec(
            name="muon",
            cls=Muon,
            kwargs={
                "lr": min(lr * 2, 0.02),
                "momentum": 0.95,
                "nesterov": True,
            },
            kind="first_order",
        ),
    ]

def sgdm_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """
    SGDM, momentum=0.9
    """
    return [
        OptimizerSpec(
            name="sgdm",
            cls=SGDM,
            kwargs={"lr": lr, "momentum": 0.9},
            kind="first_order",
        ),
    ]


def muon_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    del lr
    return [
        OptimizerSpec(
            name="muon",
            cls=Muon,
            kwargs={
                "lr": 0.02,
                "momentum": 0.95,
                "nesterov": True,
                "weight_decay": 0.0,
                "orth_after_projection": True,
            },
            kind="first_order",
        ),
    ]


def images_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """
    SGDM + Muon
    """
    return [
        OptimizerSpec(
            name="sgdm",
            cls=SGDM,
            kwargs={"lr": min(lr, 1e-3), "momentum": 0.9},
            kind="first_order",
        ),
        OptimizerSpec(
            name="muon",
            cls=Muon,
            kwargs={
                "lr": min(lr * 2, 0.02),
                "momentum": 0.95,
                "nesterov": True,
            },
            kind="first_order",
        ),
    ]

def paper_runner_config(
    *,
    steps: int,
    device: str | torch.device,
    seed: int = 42,
    log_every: int = 50,
    show_progress: bool = True,
    compile_model: bool = False,
    compile_mode: str | None = None,
) -> RunnerConfig:
    return RunnerConfig(
        steps=steps,
        device=device,
        dtype=torch.float32,
        seed=seed,
        log_every=log_every,
        chi_ema_factor=0.9,
        chi_k_full_batch_every=log_every,
        include_baseline=True,
        fail_fast=False,
        save_dir=None,
        keep_models=True,
        show_progress=show_progress,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )


ModeName = str


def resolve_projector_solver(
    projector_solver: str,
    device: str | torch.device,
) -> str:
    if projector_solver != "auto":
        return projector_solver
    return "cola_lanczos" if str(device) == "cuda" else "eigsh"


def build_run_plan(
    task_name: str,
    mode: ModeName,
    *,
    device: str | torch.device,
    steps_override: int | None = None,
    seed: int = 42,
    log_every: int = 50,
    show_progress: bool = True,
    compile_model: bool = False,
    compile_mode: str | None = None,
    projector_solver: str = "auto",
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    projector_maxiter: int | None = None,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> tuple[TaskSpec, list[OptimizerSpec], list[ProjectorSpec], RunnerConfig]:
    if task_name not in PAPER_TASKS:
        raise ValueError(
            f"Unknown task {task_name!r}. Available: {sorted(PAPER_TASKS)}"
        )

    cfg = PAPER_TASKS[task_name]
    task = cfg["task_factory"]()
    k = cfg["k"]
    lr = cfg["lr"]
    steps = steps_override if steps_override is not None else cfg["steps"]

    if mode == "paper":
        opts = paper_optimizer_specs(lr)
    elif mode == "extended":
        opts = extended_optimizer_specs(lr)
    elif mode == "transformer":
        opts = transformer_optimizer_specs(lr)
    elif mode == "image":
        opts = images_optimizer_specs(lr)
    elif mode == "sgdm":
        opts = sgdm_optimizer_specs(lr)
    elif mode == "muon":
        opts = muon_optimizer_specs(lr)
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'paper', 'extended', 'transformer', "
            f"'image', 'sgdm' or 'muon'."
        )

    solver = resolve_projector_solver(projector_solver, device)
    projs = [
        paper_projector_spec(
            k=k,
            seed=seed,
            solver=solver,
            update_every_steps=update_every_steps,
            basis_subsample=basis_subsample,
            maxiter=projector_maxiter,
            switch_on_alignment_ema=switch_on_alignment_ema,
            switch_on_step=switch_on_step,
        )
    ]
    runner_cfg = paper_runner_config(
        steps=steps,
        device=device,
        seed=seed,
        log_every=log_every,
        show_progress=show_progress,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )
    return task, opts, projs, runner_cfg
