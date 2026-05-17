from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


_REQUIRED_COLUMNS = {"optimizer", "projector", "projection", "step"}


def make_run_label(projector: str, projection: str) -> str:
    if projector == "none" and projection == "none":
        return "baseline"
    return f"{projector}:{projection}"


def run_label_sort_key(label: str) -> tuple[int, int, str]:
    lower = label.lower()

    if label == "baseline":
        family = 0
    elif "random" in lower:
        family = 1
    elif "hessian" in lower:
        family = 2
    else:
        family = 9

    if lower.endswith(":dom"):
        mode = 0
    elif lower.endswith(":bulk"):
        mode = 1
    elif lower.endswith(":none"):
        mode = 2
    else:
        mode = 9

    return family, mode, label


def _validate_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    missing = sorted((_REQUIRED_COLUMNS | {metric}) - set(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        raise ValueError(f"Column {metric!r} has no non-null values to plot.")

    plot_df["run_label"] = [
        make_run_label(str(projector), str(projection))
        for projector, projection in zip(plot_df["projector"], plot_df["projection"])
    ]
    return plot_df


def plot_metric_by_optimizer(
    df: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    optimizers: Sequence[str] | None = None,
    max_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (6.0, 4.0),
    marker: str | None = "o",
    linewidth: float = 1.5,
    markersize: float = 3.0,
    grid: bool = True,
    legend: bool = True,
    save_path: str | Path | None = None,
):
    if max_cols <= 0:
        raise ValueError("max_cols must be positive.")

    plot_df = _validate_dataframe(df, metric)

    if optimizers is None:
        optimizer_names = list(dict.fromkeys(plot_df["optimizer"].astype(str).tolist()))
    else:
        optimizer_names = list(optimizers)
        plot_df = plot_df[plot_df["optimizer"].astype(str).isin(optimizer_names)]
        if plot_df.empty:
            raise ValueError("No rows left after filtering by optimizers.")

    n_optimizers = len(optimizer_names)
    ncols = min(max_cols, n_optimizers)
    nrows = math.ceil(n_optimizers / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for ax, optimizer_name in zip(axes_flat, optimizer_names):
        opt_df = plot_df[plot_df["optimizer"].astype(str) == optimizer_name]
        if opt_df.empty:
            ax.set_title(f"{optimizer_name} (no data)")
            ax.axis("off")
            continue

        labels = sorted(opt_df["run_label"].unique(), key=run_label_sort_key)

        for label in labels:
            part = opt_df[opt_df["run_label"] == label].sort_values("step")
            ax.plot(
                part["step"],
                part[metric],
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                label=label,
            )

        ax.set_title(optimizer_name)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel or metric)
        if grid:
            ax.grid(True)
        if legend:
            ax.legend(fontsize=8)

    for ax in axes_flat[n_optimizers:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes


def plot_loss_and_accuracy_by_optimizer(
    df: pd.DataFrame,
    *,
    optimizers: Sequence[str] | None = None,
    max_cols: int = 3,
    save_dir: str | Path | None = None,
):
    save_dir_path = None if save_dir is None else Path(save_dir)

    loss_fig, loss_axes = plot_metric_by_optimizer(
        df,
        metric="loss",
        title="Loss by optimizer",
        ylabel="loss",
        optimizers=optimizers,
        max_cols=max_cols,
        save_path=None
        if save_dir_path is None
        else save_dir_path / "loss_by_optimizer.png",
    )

    acc_fig = acc_axes = None
    if "accuracy" in df.columns:
        acc_fig, acc_axes = plot_metric_by_optimizer(
            df,
            metric="accuracy",
            title="Accuracy by optimizer",
            ylabel="accuracy",
            optimizers=optimizers,
            max_cols=max_cols,
            save_path=None
            if save_dir_path is None
            else save_dir_path / "accuracy_by_optimizer.png",
        )

    return {
        "loss": (loss_fig, loss_axes),
        "accuracy": None if acc_fig is None else (acc_fig, acc_axes),
    }
