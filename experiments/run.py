"""CLI entrypoint for Song et al. (2025) replication + extension experiments.

Examples
--------

Replicate the paper (SGD baseline + Dom/Bulk-SGD, full-dataset Hessian,
basis refreshed every step, switch on EMA(chi_k) > 0.95):

    python -m experiments.run --task mnist_mlp3 --mode paper
    python -m experiments.run --task cifar10_cnn3 --mode paper
    python -m experiments.run --task sst2_transformer --mode paper

Same projector setup but extended to Muon / MeZO / ForwardGradient:

    python -m experiments.run --task mnist_mlp3 --mode extended

Run all three tasks back to back:

    python -m experiments.run --task all --mode paper

Inspect:

    mlflow ui --backend-store-uri ./mlruns
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
        choices=["paper", "extended", "transformer", "image", "sgdm", "adam", "muon", "mezo"],
        help=(
            "'paper' runs SGD only; 'extended' adds Muon/MeZO/FGD; "
            "'transformer' runs Adam/Muon; 'image' runs SGDM/Muon; "
            "'sgdm' runs SGDM only (lr from task, momentum=0.9); "
            "'adam' runs Adam only; "
            "'muon' runs Muon only (lr=0.02, momentum=0.95, orth_after_projection=False); "
            "'mezo' runs MeZO only (lr=1e-4, eps=1e-3) with Hessian dom/bulk."
        ),
    ),
    parser.add_argument(
        "--proj-mode",
        required=True,
        choices=[
            "hessian",
            "hessian_topk",
            "adaptive_lr_second_moment",
            "adaptive_lr_full_update",
            "adaptive_lr_coordinate",
            "all",
            "momentum_svd",
            "momentum-svd",
        ],
        help=(
            "'hessian' Hessian top-k eigenspace projector; "
            "'adaptive_lr_second_moment' and 'adaptive_lr_full_update' use "
            "optimizer-update statistics; 'adaptive_lr_coordinate' selects the "
            "top-k coordinates by largest effective Adam LR (1/sqrt(v)); "
            "'all' runs the paper projector set; "
            "'momentum_svd' uses optimizer momentum SVD, with layout selected "
            "by --momentum-svd-scope."
        ),
    ),
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the per-task step count from configs.PAPER_TASKS.",
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
        help="MLflow experiment name. Defaults to song2025-{task}-{mode}.",
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
            "to explain 90% of squared singular-value energy. "
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
    """Allow plain paths; mlflow expects file:// for local stores."""
    if arg.startswith(("file://", "http://", "https://", "sqlite://", "databricks")):
        return arg
    abspath = Path(arg).resolve()
    abspath.mkdir(parents=True, exist_ok=True)
    return f"file://{abspath}"


def run_one_task(
    task_name: str,
    mode: str,
    args: argparse.Namespace,
) -> None:
    device = _resolve_device(args.device)
    experiment_name = args.experiment_name or f"song2025-{task_name}-{mode}-{args.proj_mode}"

    uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(_resolve_mlflow_uri(uri))
    mlflow.set_experiment(experiment_name)

    task, opts, projs, runner_cfg = build_run_plan(
        task_name=task_name,
        mode=mode,
        projection_mode=args.proj_mode,
        device=device,
        steps_override=args.steps,
        seed=args.seed,
        log_every=args.log_every,
        show_progress=not args.no_progress,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        projector_solver=args.projector_solver,
        update_every_steps=args.update_every_steps,
        basis_subsample=args.basis_subsample,
        projector_maxiter=args.projector_maxiter,
        switch_on_alignment_ema=args.switch_on_alignment_ema,
        switch_on_step=args.switch_on_step,
        momentum_state_key=args.momentum_state_key,
        momentum_projection_type=args.momentum_projection_type,
        momentum_svd_scope=args.momentum_svd_scope,
        momentum_rank_mode=args.momentum_rank_mode,
        momentum_rank_frac=args.momentum_rank_value,
        momentum_max_rank=args.momentum_max_rank,
    )

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
