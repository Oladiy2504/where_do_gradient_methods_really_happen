"""Experiment configurations for replicating Song et al. (ICLR 2025).

Paper settings (Appendix B):
- MLP / MNIST-5k:           lr=0.01,  bs=50, MSE,  k=10, steps=20000
- CNN / CIFAR10-5k:         lr=0.001, bs=50, MSE,  k=10, steps=20000
- Transformer / SST2-1k:    lr=0.001, bs=50, MSE,  k=2,  steps=20000

Section 3.2 of the paper:
- Hessian top-k is recomputed every step
- The Hessian is the *training-loss* Hessian, i.e. averaged over the full
  training set
- SGD -> Dom/Bulk-SGD switch is triggered when EMA(chi_k) > 0.95 (alpha=0.9)
- float32 throughout

Two modes are provided here:
- "paper":    only SGD (baseline + Hessian Dom/Bulk), exact paper protocol
- "extended": adds Muon, MeZO, ForwardGradient with the same projector setup
"""

from __future__ import annotations

import torch

from src.experiments.runner import OptimizerSpec, ProjectorSpec, RunnerConfig, TaskSpec
from src.experiments.tasks import (
    make_cifar_cnn3_task,
    make_fineweb_gpt_task,
    make_mnist_mlp3_task,
    make_sst2_transformer_task,
)
from src.optimizers import SGD, SGDM, Adam, ForwardGradient, MeZO, Muon
from src.projections import (
    AdaptiveLRCoordinateProjector,
    AdaptiveLRFullUpdateProjector,
    AdaptiveLRSecondMomentProjector,
    GlobalMomentumSVDProjector,
    HessianEigenspaceProjector,
    LayerwiseMomentumSVDProjector,
    update_momentum_matrix_projector,
)


# ----------------------------- Task presets --------------------------------- #

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
    # GPT baseline (openai/parameter-golf) on FineWeb sp1024, next-token CE.
    # NOTE: this task uses the full-dataset Hessian protocol like the others,
    # but FineWeb is ~100M tokens/shard, so the basis build OOMs unless you pass
    # `--basis-subsample N` (e.g. 32-64). `--mode transformer` (Adam+Muon) is
    # the natural fit; `--mode paper` (SGD) also runs but is weak for an LM.
    "fineweb_gpt": {
        "task_factory": lambda: make_fineweb_gpt_task(
            seq_len=256,
            train_batch_tokens=16384,
            num_shards=1,
        ),
        "k": 10,
        "lr": 0.001,
        "steps": 2000,
    },
}


# ----------------------------- Projector preset ----------------------------- #

def paper_projector_spec(
    k: int,
    seed: int = 0,
    solver: str = "eigsh",
    *,
    projector_type: str = "hessian_topk",
    update_every_steps: int = 1,
    basis_subsample: int | None = None,
    maxiter: int | None = None,
    switch_on_alignment_ema: float = 0.95,
    switch_on_step: int | None = None,
) -> ProjectorSpec:
    """Projector spec matching the common Song et al. runner protocol.

    - basis recomputed every step (`update_every_steps=1`)
    - basis vectors computed over the full training set (`basis_full_dataset=True`)
    - SGD->Dom/Bulk switch triggered by EMA(chi_k) > 0.95

    `projector_type` selects the basis source; all projectors use the same
    scheduling and dom/bulk run protocol.

    For `hessian_topk`, `solver` is normally resolved by `build_run_plan(...)` to the
    device-appropriate default ("cola_lanczos" on CUDA, "eigsh" on
    CPU/MPS) — see `resolve_projector_solver`. Direct callers can override
    with `"eigsh"` or `"cola_lanczos"` explicitly.

    Tuning knobs (defaults preserve the paper protocol):
    - `update_every_steps`: how often to refresh `Q` (paper: 1).
    - `basis_subsample`: if set, truncates the full-dataset basis batch to
      the first N samples — useful for speeding up basis refresh.
    - `maxiter`: hard cap on Lanczos/ARPACK iterations. None keeps the
      solver-specific default (eigsh: scipy default; cola_lanczos:
      `min(max(4*k, 40), n-1)`).
    """
    if projector_type == "hessian_topk":
        cls = HessianEigenspaceProjector
        kwargs: dict = {"k": k, "solver": solver, "tol": 1e-4, "seed": seed}
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
            "projection_type": projection_type,
        }
    elif scope == "layerwise":
        cls = LayerwiseMomentumSVDProjector
        kwargs = {
            "k": k,
            "state_key": state_key,
            "projection_type": projection_type,
            "rank_mode": rank_mode,
            "rank_frac": rank_frac,
            "max_rank": max_rank,
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


# ----------------------------- Optimizer presets ---------------------------- #

def paper_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """Only plain SGD, as in the paper's main result figures."""
    return [
        OptimizerSpec(name="sgd", cls=SGD, kwargs={"lr": lr}, kind="first_order"),
    ]


def extended_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """SGD + Adam + Muon + MeZO + ForwardGradient with sensible defaults.

    Learning rates for MeZO/FGD are intentionally smaller because their
    gradient estimators have higher variance.
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


def adam_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """Adam only, using the task learning rate capped at 1e-3."""
    return [
        OptimizerSpec(
            name="adam",
            cls=Adam,
            kwargs={"lr": min(lr, 1e-3)},
            kind="first_order",
        ),
    ]


def transformer_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """SGD + Adam + Muon only."""
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

def mezo_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """MeZO only, same defaults as the extended preset.

    Zeroth-order: the Hessian top-k basis is still built via the loss
    closure (autograd double-backward), independent of MeZO's forward-only
    SPSA update, so dom/bulk projection applies cleanly to MeZO's update.
    lr/eps are kept small because the SPSA estimator is high-variance."""
    del lr  # intentionally unused: MeZO LR is decoupled from task LR.
    return [
        OptimizerSpec(
            name="mezo",
            cls=MeZO,
            kwargs={"lr": 1e-4, "eps": 1e-3, "seed": 0},
            kind="mezo",
        ),
    ]


def sgdm_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """SGDM only, lr from task config, momentum=0.9."""
    return [
        OptimizerSpec(
            name="sgdm",
            cls=SGDM,
            kwargs={"lr": lr, "momentum": 0.9},
            kind="first_order",
        ),
    ]


def muon_optimizer_specs(lr: float) -> list[OptimizerSpec]:
    """Muon only. lr/momentum are Muon-standard and decoupled from task lr
    (post-Newton-Schulz the update has spectral norm O(1)).
    `orth_after_projection=False` keeps the applied update inside Q for
    dom/bulk runs so chi_k/alignment describe the real update."""
    del lr  # intentionally unused: Muon LR is decoupled from task LR.
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
    """SGD + SGDM + Muon only."""
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


# ----------------------------- Runner config -------------------------------- #

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
    """RunnerConfig matching the paper: float32, chi_ema_factor=0.9."""
    return RunnerConfig(
        steps=steps,
        device=device,
        dtype=torch.float32,
        seed=seed,
        log_every=log_every,
        chi_ema_factor=0.9,
        # Paper Section 3.2: chi_k uses the full-batch gradient. Recomputing
        # every step is prohibitively expensive on the larger tasks, so we
        # refresh it on the log cadence (also the EMA/switch cadence).
        chi_k_full_batch_every=log_every,
        include_baseline=True,
        fail_fast=False,
        save_dir=None,
        keep_models=True,
        show_progress=show_progress,
        compile_model=compile_model,
        compile_mode=compile_mode,
    )


# ------------------------- Plan-building entry point ------------------------ #

ModeName = str  # "paper" | "extended"


def resolve_projector_solver(
    projector_solver: str,
    device: str | torch.device,
) -> str:
    """Map the CLI/API "auto" sentinel to a concrete solver.

    Rule (measured on this repo's paper-replication models):
      - CUDA: "cola_lanczos" — Lanczos on GPU in fp32 wins big (paper
        `mnist_mlp3 / sgd / hessian_topk / dom` drops ~4 h -> ~75 min);
      - CPU / MPS: "eigsh" — ARPACK's implicit-restart Lanczos avoids
        explicit re-orthogonalization, which cola_lanczos can't match
        without GPU parallelism behind it.
    """
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
    momentum_state_key: str = "momentum_buffer",
    momentum_projection_type: str = "two_sided",
    momentum_svd_scope: str = "global",
    momentum_rank_mode: str = "fixed",
    momentum_rank_frac: float | None = 0.05,
    momentum_max_rank: int | None = None,
) -> tuple[TaskSpec, list[OptimizerSpec], list[ProjectorSpec], RunnerConfig]:
    """Resolves a (task_name, mode) tuple into runner ingredients."""
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
    elif mode == "adam":
        opts = adam_optimizer_specs(lr)
    elif mode == "transformer":
        opts = transformer_optimizer_specs(lr)
    elif mode == "image":
        opts = images_optimizer_specs(lr)
    elif mode == "sgdm":
        opts = sgdm_optimizer_specs(lr)
    elif mode == "muon":
        opts = muon_optimizer_specs(lr)
    elif mode == "mezo":
        opts = mezo_optimizer_specs(lr)
        # MeZO is zeroth-order/high-variance: needs many more steps, a cheaper
        # basis-refresh cadence, and a step-triggered dom->bulk switch (the
        # alignment EMA is too noisy under SPSA). CLI flags still win when the
        # user passes a non-default value.
        if steps_override is None:
            steps = 300_000
        if update_every_steps == 1:
            update_every_steps = 4
        if switch_on_alignment_ema == 0.95:
            switch_on_alignment_ema = 0.9
        if switch_on_step is None:
            switch_on_step = 150_000
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Use 'paper', 'extended', 'transformer', "
            f"'image', 'sgdm', 'adam', 'muon' or 'mezo'."
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
        ]
    else:
        raise ValueError(
            "Unknown projection_mode "
            f"{projection_mode!r}. Use 'hessian', 'adaptive_lr_second_moment', "
            "'adaptive_lr_full_update', 'adaptive_lr_coordinate', 'all', or "
            "'momentum_svd'."
        )
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
