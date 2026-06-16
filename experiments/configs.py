"""Сборка задач, оптимизаторов и проекторов для основных прогонов."""

from __future__ import annotations

import torch

from src.experiments.runner import OptimizerSpec, ProjectorSpec, RunnerConfig, TaskSpec
from src.experiments.tasks import (
    make_cifar_cnn3_task,
    make_fineweb_gpt_task,
    make_mnist_mlp3_task,
    make_sst2_transformer_task,
)

from src.optimizers import SGD, SGDM, Adam, AdamW, ForwardGradient, MeZO, Muon, SubZero, SWA

from src.projections import (
    AdaptiveHessianEigenspaceProjector,
    AdaptiveLRCoordinateProjector,
    AdaptiveLRFullUpdateProjector,
    AdaptiveLRSecondMomentProjector,
    GlobalMomentumSVDProjector,
    HessianEigenspaceProjector,
    LayerwiseMomentumSVDProjector,
    MuonMetricHessianProjector,
    SpectralHessianProjector,
    update_momentum_matrix_projector,
    StiefelProjector,
    update_stiefel_projector_from_optimizer_update,
)


PAPER_TASKS: dict[str, dict] = {
    "mnist_mlp3": {
        "task_factory": lambda: make_mnist_mlp3_task(
            batch_size=50,
            loss_type="mse",
            num_classes=10,
        ),
        "k": 10,
        "lr": 0.01,
        "steps": 20000
    },
    "cifar10_cnn3": {
        "task_factory": lambda: make_cifar_cnn3_task(
            batch_size=50,
            loss_type="mse",
            num_classes=10,
        ),
        "k": 10,
        "lr": 0.001,
        "steps": 20000
    },
    "sst2_transformer": {
        "task_factory": lambda: make_sst2_transformer_task(
            batch_size=50,
            loss_type="mse",
            num_classes=2,
        ),
        "k": 2,
        "lr": 0.001,
        "steps": 20000
    },

    "fineweb_gpt": {
        # FineWeb слишком велик для полного HVP без сэмплирования basis_batch.
        "task_factory": lambda: make_fineweb_gpt_task(
            seq_len=256,
            train_batch_tokens=16384,
            num_shards=1,
        ),
        "k": 10,
        "lr": 0.001,
        "steps": 2000
    }
}


def paper_projector_spec(
    k: int,
    seed: int = 0,
    solver: str = "eigsh",
    *,
    projector_type: str = "hessian_topk",
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    maxiter: int | None = None,
    metric_eps: float = 0.1,
    metric_whiten_projection: bool = True,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> ProjectorSpec:
    """Общий протокол проекторов: full-batch basis, dom/bulk и switch по chi_k."""
    if projector_type == "hessian_topk":
        cls = HessianEigenspaceProjector
        kwargs: dict = {"k": k, "solver": solver, "tol": 1e-4, "seed": seed}
        if maxiter is not None:
            kwargs["maxiter"] = maxiter
    elif projector_type == "adaptive_hessian_topk":
        cls = AdaptiveHessianEigenspaceProjector
        kwargs = {"k": k, "solver": solver, "tol": 1e-4, "seed": seed}
        if maxiter is not None:
            kwargs["maxiter"] = maxiter
    elif projector_type == "adaptive_lr_second_moment":
        cls = AdaptiveLRSecondMomentProjector
        kwargs = {"k": k, "seed": seed}
    elif projector_type == "adaptive_lr_full_update":
        cls = AdaptiveLRFullUpdateProjector
        kwargs = {"k": k, "seed": seed}
    elif projector_type == "adaptive_lr_coordinate":
        cls = AdaptiveLRCoordinateProjector
        kwargs = {"k": k, "seed": seed}
    elif projector_type == "spectral_hessian":
        # Двусторонний Hessian-проектор для матричных параметров Muon
        cls = SpectralHessianProjector
        kwargs = {"k": k, "which": "LA", "tol": 1e-4, "seed": seed}
        if maxiter is not None:
            kwargs["inner_maxiter"] = maxiter
    elif projector_type == "muon_metric_hessian":
        # Та же идея, но в кронекеровой метрике, где живет Muon update
        cls = MuonMetricHessianProjector
        kwargs = {
            "k": k, "which": "LA", "tol": 1e-4, "seed": seed,
            "eps": metric_eps, "whiten_projection": metric_whiten_projection
        }
        if maxiter is not None:
            kwargs["inner_maxiter"] = maxiter
    else:
        raise ValueError(f"Unknown projector_type: {projector_type!r}")

    return ProjectorSpec(
        name=projector_type,
        cls=cls,
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


def momentum_svd_projector_spec(
    k: int,
    *,
    state_key: str = "momentum_buffer",
    projection_type: str = "two_sided",
    scope: str = "global",
    rank_mode: str = "fixed",
    rank_frac: float | None = 0.05,
    max_rank: int | None = None,
    update_every_steps: int = 1,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> ProjectorSpec:
    if projection_type == "tangent_rand" and scope != "layerwise":
        raise ValueError("projection_type='tangent_rand' is only supported with scope='layerwise'.")

    if scope == "global":
        cls = GlobalMomentumSVDProjector
        kwargs = {
            "k": k,
            "state_key": state_key,
            "projection_type": projection_type
        }
    elif scope == "layerwise":
        cls = LayerwiseMomentumSVDProjector
        kwargs = {
            "k": k,
            "state_key": state_key,
            "projection_type": projection_type,
            "rank_mode": rank_mode,
            "rank_frac": rank_frac,
            "max_rank": max_rank
        }
    else:
        raise ValueError(f"Unknown momentum SVD scope: {scope!r}.")

    modes = ("dom",) if projection_type == "tangent_rand" else ("dom", "bulk")

    return ProjectorSpec(
        name=f"momentum_svd_{scope}_{projection_type}",
        cls=cls,
        kwargs=kwargs,
        modes=modes,
        update_kind="custom",
        update_before_train=False,
        update_every_steps=update_every_steps,
        update_fn=update_momentum_matrix_projector,
        basis_full_dataset=False,
        switch_on_alignment_ema=switch_on_alignment_ema,
        switch_on_step=switch_on_step,
    )


def stiefel_projector_spec(
    k: int,
    seed: int = 0,
    *,
    lr: float = 1e-2,
    retraction_method: str = "cayley",
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> ProjectorSpec:
    return ProjectorSpec(
        name=f"stiefel_{retraction_method}_{lr}",
        cls=StiefelProjector,
        kwargs={
            "k": k,
            "lr": lr,
            "seed": seed,
            "retraction_method": retraction_method
        },
        modes=("dom", "bulk"),
        update_kind="custom",
        update_before_train=True,
        update_every_steps=update_every_steps,
        update_fn=update_stiefel_projector_from_optimizer_update,
        basis_full_dataset=True,
        basis_subsample=basis_subsample,
        switch_on_alignment_ema=switch_on_alignment_ema,
        switch_on_step=switch_on_step,
    )


def paper_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    return [
        OptimizerSpec(name="sgd", cls=SGD, kwargs={"lr": lr}, kind="first_order")
    ]


def extended_optimizer_specs(lr: float) -> list[OptimizerSpec]:
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
        OptimizerSpec(
            name="swa",
            cls=SWA,
            kwargs={"lr": lr, "momentum": 0.9, "swa_start": 0, "swa_freq": 1},
            kind="first_order",
        )
    ]


def adam_optimizer_specs(
    lr: float, beta1: float = 0.9, *, clamp_lr: bool = True
) -> list[OptimizerSpec]:
    return [
        OptimizerSpec(
            name="adam" if beta1 == 0.9 else f"adam-b1{beta1:g}",
            cls=Adam,
            kwargs={"lr": min(lr, 1e-3) if clamp_lr else lr, "betas": (beta1, 0.999)},
            kind="first_order",
        )
    ]


def adamw_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    return [
        OptimizerSpec(
            name="adamw",
            cls=AdamW,
            kwargs={"lr": min(lr, 1e-3), "weight_decay": 0.01},
            kind="first_order",
        )
    ]


def transformer_optimizer_specs(lr: float) -> list[OptimizerSpec]:
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
                "nesterov": True
            },
            kind="first_order",
        )
    ]


def mezo_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """MeZO держим отдельно: шаг задачи здесь слишком крупный для SPSA-оценки."""
    del lr
    return [
        OptimizerSpec(
            name="mezo",
            cls=MeZO,
            kwargs={"lr": 1e-4, "eps": 1e-3, "seed": 0},
            kind="mezo",
        )
    ]


def forward_gradient_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """ForwardGradient тоже использует свой малый lr из-за дисперсии оценки."""
    del lr
    return [
        OptimizerSpec(
            name="forward_gradient",
            cls=ForwardGradient,
            kwargs={"lr": 6e-6, "seed": 0},
            kind="forward_gradient",
        )
    ]


def subzero_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    del lr
    return [
        OptimizerSpec(
            name="subzero",
            cls=SubZero,
            kwargs={
                "lr": 1e-4,
                "eps": 1e-3,
                "seed": 0,
                "rank": 8,
                "update_freq": 100
            },
            kind="mezo",
        )
    ]


def sgdm_optimizer_specs(lr: float, momentum: float = 0.9) -> list[OptimizerSpec]:
    return [
        OptimizerSpec(
            name="sgdm" if momentum == 0.9 else f"sgdm-m{momentum:g}",
            cls=SGDM,
            kwargs={"lr": lr, "momentum": momentum},
            kind="first_order",
        )
    ]


def swa_optimizer_specs(
    lr: float,
    *,
    momentum: float = 0.9,
    swa_start: int = 0,
    swa_freq: int = 1,
    lr_min: float | None = None,
    cycle_length: int | None = None,
) -> list[OptimizerSpec]:
    if (lr_min is None) != (cycle_length is None):
        raise ValueError("lr_min and cycle_length must be set together.")

    kwargs = {
        "lr": lr,
        "momentum": momentum,
        "swa_start": swa_start,
        "swa_freq": swa_freq
    }
    if lr_min is not None:
        kwargs["lr_min"] = lr_min
        kwargs["cycle_length"] = cycle_length

    return [
        OptimizerSpec(
            name="swa",
            cls=SWA,
            kwargs=kwargs,
            kind="first_order",
        )
    ]


def muon_optimizer_specs(
    lr: float, polynom: str = "jordan", momentum: float = 0.95
) -> list[OptimizerSpec]:
    del lr
    return [
        OptimizerSpec(
            name=f"muon-{polynom}" if polynom != "jordan" else "muon",
            cls=Muon,
            kwargs={
                "lr": 0.02,
                "momentum": momentum,
                "nesterov": True,
                "weight_decay": 0.0,
                "orth_after_projection": False,
                "polynom": polynom
            },
            kind="first_order",
        )
    ]


def images_optimizer_specs(lr: float) -> list[OptimizerSpec]:
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
                "nesterov": True
            },
            kind="first_order",
        )
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
    log_top_eigvals: int | None = None,
    stable_rank_probes: int | None = None,
    swa_from_step: int | None = None,
    frozen_bulk: bool = False,
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
        log_top_eigvals=log_top_eigvals,
        stable_rank_probes=stable_rank_probes,
        swa_from_step=swa_from_step,
        frozen_bulk=frozen_bulk,
    )

ModeName = str


def resolve_projector_solver(
    projector_solver: str,
    device: str | torch.device,
) -> str:
    """На CUDA выгоднее cola_lanczos, на CPU/MPS надежнее scipy eigsh."""
    if projector_solver != "auto":
        return projector_solver
    return "cola_lanczos" if str(device) == "cuda" else "eigsh"


def build_run_plan(
    task_name: str,
    mode: ModeName,
    *,
    projection_mode: str = "hessian",
    device: str | torch.device,
    steps_override: int | None = None,
    lr_override: float | None = None,
    seed: int = 42,
    log_every: int = 50,
    show_progress: bool = True,
    compile_model: bool = False,
    compile_mode: str | None = None,
    projector_solver: str = "auto",
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    projector_maxiter: int | None = None,
    metric_eps: float = 0.1,
    metric_whiten_projection: bool = True,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
    experiment: str = "switch",
    skip_none: bool = False,
    skip_dom: bool = False,
    skip_bulk: bool = False,
    swa_from_step: int | None = None,
    frozen_bulk: bool = False,
    num_eigvals: int = 20,
    stable_rank_probes: int = 16,
    adam_beta1: float = 0.9,
    sgdm_momentum: float = 0.9,
    muon_polynom: str = "jordan",
    muon_momentum: float = 0.95,
    swa_momentum: float = 0.9,
    swa_start: int = 0,
    swa_freq: int = 1,
    swa_lr_min: float | None = None,
    swa_cycle_length: int | None = None,
    momentum_state_key: str = "momentum_buffer",
    momentum_projection_type: str = "two_sided",
    momentum_svd_scope: str = "global",
    momentum_rank_mode: str = "fixed",
    momentum_rank_frac: float | None = 0.05,
    momentum_max_rank: int | None = None,
    stiefel_retraction: str = "cayley",
    stiefel_lr: float = 1e-2,
) -> tuple[TaskSpec, list[OptimizerSpec], list[ProjectorSpec], RunnerConfig]:
    if task_name not in PAPER_TASKS:
        raise ValueError(
            f"Unknown task {task_name!r}. Available: {sorted(PAPER_TASKS)}"
        )

    cfg = PAPER_TASKS[task_name]
    task = cfg["task_factory"]()
    k = cfg["k"]
    lr = lr_override if lr_override is not None else cfg["lr"]
    steps = steps_override if steps_override is not None else cfg["steps"]

    if mode == "paper":
        opts = paper_optimizer_specs(lr)
    elif mode == "extended":
        opts = extended_optimizer_specs(lr)
    elif mode == "adam":
        opts = adam_optimizer_specs(lr, beta1=adam_beta1, clamp_lr=lr_override is None)
    elif mode == "adamw":
        opts = adamw_optimizer_specs(lr)
    elif mode == "transformer":
        opts = transformer_optimizer_specs(lr)
    elif mode == "image":
        opts = images_optimizer_specs(lr)
    elif mode == "sgdm":
        opts = sgdm_optimizer_specs(lr, momentum=sgdm_momentum)
    elif mode == "swa":
        opts = swa_optimizer_specs(lr, momentum=swa_momentum, swa_start=swa_start, swa_freq=swa_freq,
                                   lr_min=swa_lr_min, cycle_length=swa_cycle_length)
    elif mode == "muon":
        opts = muon_optimizer_specs(lr, polynom=muon_polynom, momentum=muon_momentum)
    elif mode == "mezo":
        opts = mezo_optimizer_specs(lr)
        if steps_override is None:
            steps = 300_000
        if update_every_steps == 1:
            update_every_steps = 10
        if switch_on_alignment_ema == 0.95:
            switch_on_alignment_ema = 0.9
        if switch_on_step is None:
            switch_on_step = 150_000
    elif mode == "forward_gradient":
        opts = forward_gradient_optimizer_specs(lr)
        if steps_override is None:
            steps = 300_000
        if update_every_steps == 1:
            update_every_steps = 100

        if switch_on_alignment_ema == 0.95:
            switch_on_alignment_ema = 0.9
        if switch_on_step is None:
            switch_on_step = 150_000
    elif mode == "subzero":
        opts = subzero_optimizer_specs(lr)
        if steps_override is None:
            steps = 300_000
        if update_every_steps == 1:
            update_every_steps = 10
        if switch_on_alignment_ema == 0.95:
            switch_on_alignment_ema = 0.9
        if switch_on_step is None:
            switch_on_step = 150_000
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'paper', 'extended', 'transformer', "
            f"'image', 'sgdm', 'swa', 'adam', 'adamw', 'muon', 'mezo', "
            f"'subzero' or 'forward_gradient'."
        )

    solver = resolve_projector_solver(projector_solver, device)
    projection_mode = projection_mode.replace("-", "_")
    if projection_mode in ("hessian", "hessian_topk"):
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="hessian_topk",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode in ("adaptive_hessian", "adaptive_hessian_topk"):
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_hessian_topk",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "spectral_hessian":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="spectral_hessian",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "muon_metric_hessian":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="muon_metric_hessian",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                metric_eps=metric_eps,
                metric_whiten_projection=metric_whiten_projection,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "adaptive_lr_second_moment":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_second_moment",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "adaptive_lr_full_update":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_full_update",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "adaptive_lr_coordinate":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_coordinate",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "all":
        projs = [
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="hessian_topk",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            ),
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_hessian_topk",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            ),
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_second_moment",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            ),
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_full_update",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            ),
            paper_projector_spec(
                k=k,
                seed=seed,
                solver=solver,
                projector_type="adaptive_lr_coordinate",
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                maxiter=projector_maxiter,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "momentum_svd":
        projs = [
            momentum_svd_projector_spec(
                k=k,
                state_key=momentum_state_key,
                projection_type=momentum_projection_type,
                scope=momentum_svd_scope,
                rank_mode=momentum_rank_mode,
                rank_frac=momentum_rank_frac,
                max_rank=momentum_max_rank,
                update_every_steps=update_every_steps,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    elif projection_mode == "stiefel":
        projs = [
            stiefel_projector_spec(
                k=k,
                seed=seed,
                lr=stiefel_lr,
                retraction_method=stiefel_retraction,
                update_every_steps=update_every_steps,
                basis_subsample=basis_subsample,
                switch_on_alignment_ema=switch_on_alignment_ema,
                switch_on_step=switch_on_step,
            )
        ]
    else:
        raise ValueError(
            "Unknown projection_mode "
            f"{projection_mode!r}. Use 'hessian', 'adaptive_hessian', "
            "'spectral_hessian', 'muon_metric_hessian', "
            "'adaptive_lr_second_moment', "
            "'adaptive_lr_full_update', 'adaptive_lr_coordinate', 'all', or "
            "'momentum_svd', or 'stiefel'."
        )

    log_top_eigvals = None
    stable_rank_probes_cfg = None
    if experiment != "switch":
        # В режимах наблюдения тренируем raw-оптимизатор и только меряем alignment.
        HESSIAN_LIKE = {
            "hessian_topk", "adaptive_hessian_topk",
            "spectral_hessian", "muon_metric_hessian"
        }
        if experiment in ("eigenvalues", "stable_rank"):
            for spec in projs:
                if spec.name not in HESSIAN_LIKE:
                    raise ValueError(
                        f"--experiment {experiment} requires a Hessian-like "
                        f"--proj-mode ({sorted(HESSIAN_LIKE)}); got {spec.name!r}."
                    )
            if experiment == "eigenvalues":
                for spec in projs:
                    spec.kwargs = {**spec.kwargs, "k": num_eigvals}
                log_top_eigvals = num_eigvals
            else:
                stable_rank_probes_cfg = stable_rank_probes
        elif experiment != "alignment":
            raise ValueError(
                f"Unknown experiment {experiment!r}. Use 'switch', 'alignment', "
                "'eigenvalues', or 'stable_rank'."
            )
        for spec in projs:
            spec.modes = ("dom",)
            spec.switch_on_alignment_ema = None
            spec.switch_on_step = None

    runner_cfg = paper_runner_config(
        steps=steps,
        device=device,
        seed=seed,
        log_every=log_every,
        show_progress=show_progress,
        compile_model=compile_model,
        compile_mode=compile_mode,
        log_top_eigvals=log_top_eigvals,
        stable_rank_probes=stable_rank_probes_cfg,
        swa_from_step=swa_from_step,
        frozen_bulk=frozen_bulk,
    )

    if experiment != "switch":
        runner_cfg.include_baseline = False

    if skip_none:
        runner_cfg.include_baseline = False
    drop_modes = {m for m, skip in (("dom", skip_dom), ("bulk", skip_bulk)) if skip}
    if drop_modes:
        for spec in projs:
            spec.modes = tuple(m for m in spec.modes if m not in drop_modes)
        projs = [spec for spec in projs if spec.modes]
    if not runner_cfg.include_baseline and not projs:
        raise ValueError(
            "Nothing left to run: --skip-none with no remaining projector "
            "modes (check --skip-dom/--skip-bulk against --proj-mode/--experiment)."
        )

    return task, opts, projs, runner_cfg
