from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiments.configs import build_run_plan
from src.experiments.runner import ExperimentRunner, results_to_dataframe


# Edit this block before a local run.
EXPERIMENT_NAME = "momentum_cifar_adam_expavg_layerwise_tangent_energy080_switch_1000_end_10000_upd1"
EXPERIMENT_TITLE = "Adam exp_avg layerwise momentum tangent projection CIFAR run (energy 80%)"

RUN_PLAN_KWARGS = dict(
    task_name="cifar10_cnn3",
    mode="adam",
    projection_mode="momentum_svd",
    steps_override=3000,
    seed=42,
    log_every=10,
    update_every_steps=1,
    basis_subsample=None,
    switch_on_step=1000,
    projector_solver="auto",
    projector_maxiter=40,
    show_progress=True,
    momentum_projection_type="tangent",
    momentum_state_key="exp_avg",
    momentum_svd_scope="layerwise",
    momentum_rank_mode="energy",
    momentum_rank_frac=0.8,
    momentum_max_rank=None,
)

RUNNER_OVERRIDES = dict(
    keep_models=False,
    fail_fast=True,
    chi_k_full_batch_every=None,
)

PLOT_SPECS = [
    ("loss", "Loss"),
    ("accuracy", "Accuracy"),
    ("chi_k_grad", "Gradient alignment with projection subspace"),
    ("chi_k_grad_ema", "EMA of gradient alignment"),
    ("subspace_usefulness/rho", "Subspace usefulness ratio"),
    ("projector/subspace_pct", "Projection subspace size (% of parameters)"),
]


def resolve_device(arg: str = "auto") -> str:
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return arg


def plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    if metric not in df.columns or not df[metric].notna().any():
        print(f"skip plot: {metric} is absent or all NaN")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, group in df.groupby("run"):
        group = group.sort_values("step")
        ax.plot(group["step"], group[metric], label=run_name)

    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metrics_dashboard(
        df: pd.DataFrame,
        metrics: list[tuple[str, str]],
        title: str,
        out_path: Path,
) -> None:
    available = [
        (metric, label)
        for metric, label in metrics
        if metric in df.columns and df[metric].notna().any()
    ]
    if not available:
        print("skip dashboard: no requested metrics are present")
        return

    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 4.2 * nrows),
        squeeze=False,
        sharex=True,
    )

    handles = []
    labels = []
    for ax, (metric, label) in zip(axes.ravel(), available):
        for run_name, group in df.groupby("run"):
            group = group.sort_values("step")
            line, = ax.plot(group["step"], group[metric], label=run_name)
            if run_name not in labels:
                handles.append(line)
                labels.append(run_name)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(available):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 3),
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.06 if handles else 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.show()


def main() -> None:
    device = resolve_device("auto")

    task, opts, projs, runner_cfg = build_run_plan(
        device=device,
        **RUN_PLAN_KWARGS,
    )

    out_dir = Path("local_runs") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg.save_dir = out_dir
    for key, value in RUNNER_OVERRIDES.items():
        setattr(runner_cfg, key, value)

    print(f"experiment={EXPERIMENT_NAME}")
    print(f"device={device}")
    print(f"optimizers={[o.name for o in opts]}")
    print(f"projectors={[p.name for p in projs]}")
    print(f"save_dir={out_dir}")

    runner = ExperimentRunner(task, opts, projs, runner_cfg)
    results = runner.run()

    df = results_to_dataframe(results)
    df.to_csv(out_dir / "history.csv", index=False)

    display_cols = [
        "run",
        "step",
        "loss",
        "accuracy",
        "chi_k_grad",
        "chi_k_grad_ema",
        "subspace_usefulness/rho",
        "projector/subspace_pct",
        "effective_projection",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].tail(30))

    for metric, label in PLOT_SPECS:
        metric_file = metric.replace("/", "_")
        plot_metric(
            df,
            metric,
            f"{EXPERIMENT_TITLE}: {label}",
            out_dir / f"{EXPERIMENT_NAME}__{metric_file}.png",
        )

    plot_metrics_dashboard(
        df,
        PLOT_SPECS,
        EXPERIMENT_TITLE,
        out_dir / f"{EXPERIMENT_NAME}__dashboard.png",
    )

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
