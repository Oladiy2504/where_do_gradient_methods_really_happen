"""CLI для запуска экспериментов Song et al. и расширенных проверок.

Примеры:
    python -m experiments.run --task mnist_mlp3 --mode paper --proj-mode hessian
    python -m experiments.run --task cifar10_cnn3 --mode adam --proj-mode momentum_svd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
import torch
from dotenv import load_dotenv

from experiments.configs import PAPER_TASKS, build_run_plan
from experiments.mlflow_runner import MLflowLoggingRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(PAPER_TASKS) + ["all"],
        help="Task to run (or 'all' to run every paper task).",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["paper", "extended", "transformer", "image", "sgdm", "swa", "adam", "adamw", "muon", "mezo", "subzero", "forward_gradient"],
        help=(
            "'paper' runs SGD only; 'extended' adds Muon/MeZO/FGD/SWA; "
            "'transformer' runs Adam/Muon; 'image' runs SGDM/Muon; "
            "'sgdm' runs SGDM only (lr from task, momentum=0.9); "
            "'swa' runs SWA only (SGDM trajectory + equal-weight averaging); "
            "'adam' runs Adam only; "
            "'adamw' runs AdamW only (lr from task, weight_decay=0.01); "
            "'muon' runs Muon only (lr=0.02, momentum=0.95, orth_after_projection=False); "
            "'mezo' runs MeZO only (lr=1e-4, eps=1e-3) with Hessian dom/bulk; "
            "'subzero' runs SubZero only (lr=1e-4, eps=1e-3, rank=8) with Hessian dom/bulk; "
            "'forward_gradient' runs ForwardGradient only (lr=6e-6) with Hessian dom/bulk."
        ),
    )
    parser.add_argument(
        "--proj-mode",
        required=True,
        choices=[
            "hessian",
            "hessian_topk",
            "adaptive_hessian",
            "adaptive_hessian_topk",
            "spectral_hessian",
            "muon_metric_hessian",
            "adaptive_lr_second_moment",
            "adaptive_lr_full_update",
            "adaptive_lr_coordinate",
            "all",
            "momentum_svd",
            "momentum-svd",
            "stiefel"
        ],
        help=(
            "'hessian' Hessian top-k eigenspace projector; "
            "'adaptive_hessian' Hessian top-k in the adaptive geometry "
            "P^{-1/2} H P^{-1/2}, P=diag(sqrt(v)+eps); "
            "'spectral_hessian' two-sided (Kronecker) top-k Hessian subspace "
            "dom(d)=U Uᵀ d V Vᵀ via HOOI — the spectral/Muon analog of 'hessian'; "
            "'muon_metric_hessian' = 'spectral_hessian' re-metrized into Muon's "
            "Kronecker metric M=L⊗R (whitened HVP), tuned by --spectral-metric-eps; "
            "'adaptive_lr_second_moment' and 'adaptive_lr_full_update' use "
            "optimizer-update statistics; 'adaptive_lr_coordinate' selects the "
            "top-k coordinates by largest effective Adam LR (1/sqrt(v)); "
            "'all' runs the paper projector set; "
            "'momentum_svd' uses optimizer momentum SVD, with layout selected "
            "by --momentum-svd-scope; "
            "'stiefel' uses the online Stiefel projector with retraction selected "
            "by --stiefel-retraction."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the per-task step count from configs.PAPER_TASKS.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the per-task base optimizer learning rate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device. 'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help=(
            "MLflow tracking URI. Falls back to $MLFLOW_TRACKING_URI (e.g. from .env), "
            "then to a local MLflow server on http://127.0.0.1:5000."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Defaults to "
        "song2025-{task}-{mode}-{proj_mode}-{experiment}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-step progress prints.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not run the optimizer-only baseline for each optimizer/projector plan.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model in torch.compile (dynamic shapes). Off by default.",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode. Ignored unless --compile is set.",
    )
    parser.add_argument(
        "--projector-solver",
        default="auto",
        choices=["auto", "eigsh", "cola_lanczos", "gram_eigh"],
        help="Backend for the Hessian top-k eigensolver. 'auto' (default) "
        "picks cola_lanczos on CUDA and eigsh on CPU/MPS. 'eigsh' is scipy "
        "ARPACK; 'cola_lanczos' runs Lanczos on the model's device via cola-ml;" \
        "'gram_eigh' is recommended for gradient covariance projector.",
    )
    parser.add_argument(
        "--update-every-steps",
        type=int,
        default=1,
        help="How often to recompute the Hessian top-k basis Q. Paper protocol: 1.",
    )
    parser.add_argument(
        "--basis-subsample",
        type=int,
        default=None,
        help="If set, truncates the full-dataset basis batch to the first N "
        "samples. None (default) keeps the full training set.",
    )
    parser.add_argument(
        "--projector-maxiter",
        type=int,
        default=None,
        help="Hard cap on Lanczos/ARPACK iterations per basis refresh. None "
        "(default) keeps the solver-specific default.",
    )
    parser.add_argument(
        "--spectral-metric-eps",
        type=float,
        default=0.1,
        help="Damping for --proj-mode muon_metric_hessian: floors each Gram "
        "factor's spectrum at eps·λ_max when forming Muon's metric M=L⊗R, "
        "bounding the whitening's condition number to eps^-1/4 (needed because "
        "finite-batch GᵀG is rank-deficient). Smaller = more Muon-faithful but "
        "more aggressive; large values recover plain spectral_hessian.",
    )
    parser.add_argument(
        "--spectral-metric-euclidean-projection",
        action="store_true",
        help="For --proj-mode muon_metric_hessian: find the basis from the "
        "whitened (Muon-metric) Hessian, but project and measure chi_k in plain "
        "Euclidean parameter space (dom=Ũ Ũᵀ d Ṽ Ṽᵀ), exactly like "
        "spectral_hessian — so the alignment is param-space and comparable. "
        "Default (off) uses the full whitened-metric geometry.",
    )
    parser.add_argument(
        "--switch-on-alignment-ema",
        type=float,
        default=0.95,
        help="Threshold on EMA(chi_k) that flips SGD -> Dom/Bulk-SGD. Paper "
        "protocol: 0.95.",
    )
    parser.add_argument(
        "--switch-on-step",
        type=int,
        default=None,
        help="If set, forces the SGD -> Dom/Bulk switch at this train step "
        "(1-indexed). Fires the same code path as --switch-on-alignment-ema; "
        "when both trigger, the first to fire wins.",
    )
    parser.add_argument(
        "--skip-none",
        action="store_true",
        help="Skip the 'none' baseline run (the raw optimizer with no "
        "projection). Sets RunnerConfig.include_baseline=False.",
    )
    parser.add_argument(
        "--skip-dom",
        action="store_true",
        help="Skip the 'dom' (Dom-projected) run by dropping 'dom' from each "
        "projector's modes.",
    )
    parser.add_argument(
        "--skip-bulk",
        action="store_true",
        help="Skip the 'bulk' (Bulk-projected) run by dropping 'bulk' from "
        "each projector's modes.",
    )
    parser.add_argument(
        "--swa_from_step",
        type=int,
        default=None,
        help="Enable Stochastic Weight Averaging starting at this train step "
        "(1-indexed). Once step >= N the runner keeps an equal-weight running "
        "mean of the model parameters in a side buffer, logs the averaged "
        "model's metrics as swa/loss (+ swa/<metric>) at each log step, and "
        "swaps the SWA weights into the model at run end. Applies to every run "
        "(none/dom/bulk). None (default) disables SWA.",
    )
    parser.add_argument(
        "--frozen-bulk",
        action="store_true",
        help="Freeze the bulk subspace: a 'bulk' projection run builds its "
        "basis only once (at the first build) and reuses it for every step, "
        "skipping the periodic --update-every-steps refresh. 'dom' runs are "
        "unaffected. In the paired dom->bulk protocol the frozen basis is the "
        "one computed at the switch point.",
    )
    parser.add_argument(
        "--experiment",
        default="switch",
        choices=["switch", "alignment", "eigenvalues", "stable_rank"],
        help=(
            "Experiment scenario. 'switch' (default) = the SGD -> Dom/Bulk "
            "switch protocol (--switch-on-* apply). 'alignment' = train "
            "normally and only measure chi_k alignment with the dom subspace, "
            "never switching (--switch-on-* are ignored). 'eigenvalues' = like "
            "'alignment' but also log the top-N eigenvalues (--num-eigvals), "
            "setting the projector rank k=N. 'stable_rank' = like 'alignment' "
            "but also log the operator's stable rank ||H||_F^2/||H||_2^2 "
            "(--stable-rank-probes). 'eigenvalues'/'stable_rank' are Hessian-"
            "like --proj-mode only."
        ),
    )
    parser.add_argument(
        "--num-eigvals",
        type=int,
        default=20,
        help="Number of top eigenvalues to log under --experiment eigenvalues. "
        "Also sets the projector rank k=N for that run (independent of the "
        "task's #classes).",
    )
    parser.add_argument(
        "--stable-rank-probes",
        type=int,
        default=16,
        help="Number of Hutchinson probes used to estimate ||H||_F^2 for the "
        "stable rank under --experiment stable_rank. More probes = lower "
        "variance, more HVPs per log step.",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="First-moment (momentum) decay for --mode adam. Default 0.9. "
        "Pass 0 to disable momentum: the first-moment buffer becomes the raw "
        "gradient, so the update is lr*grad/(sqrt(v_hat)+eps).",
    )
    parser.add_argument(
        "--sgdm-momentum",
        type=float,
        default=0.9,
        help="Momentum for --mode sgdm. Default 0.9. Pass 0 to reduce SGDM to "
        "plain SGD. Ignored for non-sgdm modes.",
    )
    parser.add_argument(
        "--muon-polynom",
        default="jordan",
        choices=["jordan", "cans", "polarexpress"],
        help=(
            "Newton-Schulz orthogonalisation polynomial for --mode muon. "
            "'jordan' (default) is the standard quintic; 'cans' uses the "
            "optimal odd-cubic CANS iteration; 'polarexpress' uses PolarExpress. "
            "Ignored for non-muon modes."
        ),
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Momentum (EMA decay) for --mode muon. Default 0.95. Pass 0 to "
        "disable momentum: the orthogonalised direction becomes the raw "
        "gradient. Ignored for non-muon modes.",
    )
    parser.add_argument(
        "--stiefel-retraction",
        default="cayley",
        choices=["cayley", "cans"],
        help="Retraction used by --proj-mode stiefel."
    )
    parser.add_argument(
        "--stiefel-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the online Stiefel projector update."
    )
    parser.add_argument(
        "--swa-momentum",
        type=float,
        default=0.9,
        help="Momentum for --mode swa's SGDM trajectory. Default 0.9. "
        "Ignored for non-SWA modes.",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=0,
        help="First step eligible for SWA averaging. Default 0 includes the "
        "initial weights in the average, matching the SWA paper's algorithm. "
        "Ignored for non-SWA modes.",
    )
    parser.add_argument(
        "--swa-freq",
        type=int,
        default=1,
        help="Average one parameter sample every N steps for --mode swa. "
        "Default 1. Ignored for non-SWA modes.",
    )
    parser.add_argument(
        "--swa-lr-min",
        type=float,
        default=None,
        help="If set with --swa-cycle-length, enables SWA's linear cyclic LR: "
        "each cycle jumps to task lr and decreases to this value. Default None "
        "uses constant lr. Ignored for non-SWA modes.",
    )
    parser.add_argument(
        "--swa-cycle-length",
        type=int,
        default=None,
        help="Cycle length in steps for --mode swa when --swa-lr-min is set. "
        "Ignored for non-SWA modes."
    )
    parser.add_argument(
        "--momentum-state-key",
        default="momentum_buffer",
        help="Optimizer state key for --proj-mode momentum_svd. Use "
        "'momentum_buffer' for SGDM/Muon or 'exp_avg' for Adam.",
    )
    parser.add_argument(
        "--momentum-projection-type",
        default="two_sided",
        choices=["two_sided", "tangent", "tangent_rand"],
        help=(
            "Momentum-SVD projection geometry. 'two_sided' keeps only the "
            "strict U U^T Z V V^T block; 'tangent' keeps the larger tangent "
            "space U U^T Z + Z V V^T - U U^T Z V V^T; 'tangent_rand' "
            "uses the same layerwise ranks as tangent but replaces U,V with "
            "random orthonormal matrices and runs dom only."
        ),
    )
    parser.add_argument(
        "--momentum-svd-scope",
        default="global",
        choices=["global", "layerwise"],
        help=(
            "Momentum-SVD basis layout. 'global' flattens selected parameters "
            "into one artificial matrix; 'layerwise' computes a separate SVD "
            "for each matrix-shaped parameter and passes 1D params through."
        ),
    )
    parser.add_argument(
        "--momentum-rank-mode",
        default="fixed",
        choices=["fixed", "fraction", "energy"],
        help=(
            "Layerwise-only rank rule. 'fixed' uses min(task k, min(a,b)) "
            "for every layer; 'fraction' uses floor(rank_value * min(a,b)); "
            "'energy' picks the smallest rank whose squared singular values "
            "explain at least rank_value of layer momentum energy. Fraction "
            "and energy ranks can be capped by --momentum-max-rank."
        ),
    )
    parser.add_argument(
        "--momentum-rank-value",
        type=float,
        default=0.05,
        help=(
            "Layerwise-only rank value. With --momentum-rank-mode=fraction, "
            "this is the fraction of min(a,b). With "
            "--momentum-rank-mode=energy, this is the cumulative energy "
            "threshold in (0, 1], e.g. 0.9 keeps enough singular directions "
            "to explain 90%% of squared singular-value energy. "
        ),
    )
    parser.add_argument(
        "--momentum-max-rank",
        type=int,
        default=None,
        help=(
            "Optional layerwise-only upper bound on ranks produced by "
            "--momentum-rank-mode=fraction or energy. Ignored for fixed/global modes."
        ),
    )
    return parser.parse_args(argv)


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return arg


def _resolve_mlflow_uri(arg: str) -> str:
    """MLflow требует file:// для локальных путей, поэтому нормализуем их здесь."""
    if arg.startswith(("file://", "http://", "https://", "sqlite://", "databricks")):
        return arg
    abspath = Path(arg).resolve()
    abspath.mkdir(parents=True, exist_ok=True)
    return abspath.as_uri()


def run_one_task(
    task_name: str,
    mode: str,
    args: argparse.Namespace,
) -> None:
    device = _resolve_device(args.device)
    experiment_name = (
        args.experiment_name
        or f"song2025-{task_name}-{mode}-{args.proj_mode}-{args.experiment}"
    )

    uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(_resolve_mlflow_uri(uri))
    mlflow.set_experiment(experiment_name)

    task, opts, projs, runner_cfg = build_run_plan(
        task_name=task_name,
        mode=mode,
        projection_mode=args.proj_mode,
        device=device,
        steps_override=args.steps,
        lr_override=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        show_progress=not args.no_progress,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        projector_solver=args.projector_solver,
        update_every_steps=args.update_every_steps,
        basis_subsample=args.basis_subsample,
        projector_maxiter=args.projector_maxiter,
        metric_eps=args.spectral_metric_eps,
        metric_whiten_projection=not args.spectral_metric_euclidean_projection,
        switch_on_alignment_ema=args.switch_on_alignment_ema,
        switch_on_step=args.switch_on_step,
        experiment=args.experiment,
        skip_none=args.skip_none,
        skip_dom=args.skip_dom,
        skip_bulk=args.skip_bulk,
        swa_from_step=args.swa_from_step,
        frozen_bulk=args.frozen_bulk,
        num_eigvals=args.num_eigvals,
        stable_rank_probes=args.stable_rank_probes,
        adam_beta1=args.adam_beta1,
        sgdm_momentum=args.sgdm_momentum,
        muon_polynom=args.muon_polynom,
        muon_momentum=args.muon_momentum,
        swa_momentum=args.swa_momentum,
        swa_start=args.swa_start,
        swa_freq=args.swa_freq,
        swa_lr_min=args.swa_lr_min,
        swa_cycle_length=args.swa_cycle_length,
        momentum_state_key=args.momentum_state_key,
        momentum_projection_type=args.momentum_projection_type,
        momentum_svd_scope=args.momentum_svd_scope,
        momentum_rank_mode=args.momentum_rank_mode,
        momentum_rank_frac=args.momentum_rank_value,
        momentum_max_rank=args.momentum_max_rank,
        stiefel_retraction=args.stiefel_retraction,
        stiefel_lr=args.stiefel_lr,
    )
    if args.no_baseline:
        runner_cfg.include_baseline = False

    print(
        f"\n[{task_name} / {mode}] device={device}, steps={runner_cfg.steps}, "
        f"optimizers={[o.name for o in opts]}, projectors={[p.name for p in projs]}",
        flush=True,
    )

    runner = MLflowLoggingRunner(task, opts, projs, runner_cfg)
    results = runner.run()

    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"[{task_name} / {mode}] done: ok={ok}, failed={failed}", flush=True)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    if args.task == "all":
        for task_name in PAPER_TASKS:
            run_one_task(task_name, args.mode, args)
    else:
        run_one_task(args.task, args.mode, args)

    return 0

if __name__ == "__main__":
    sys.exit(main())
