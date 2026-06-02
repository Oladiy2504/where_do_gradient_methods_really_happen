from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.configs import build_run_plan
from src.experiments.runner import ExperimentRunner, results_to_dataframe


EXPERIMENT_NAME = "momentum_cifar_adam_expavg_layerwise_tangent_rand_energy080_switch_1000_end_3000_upd1_3seeds"
EXPERIMENT_TITLE = "Adam exp_avg layerwise random tangent CIFAR (energy-matched 80%) sanity-check"
SEEDS = [42, 67, 112]

RUN_PLAN_KWARGS = dict(
    task_name="cifar10_cnn3",
    mode="adam",
    projection_mode="momentum_svd",
    steps_override=3000,
    log_every=20,
    update_every_steps=1,
    basis_subsample=None,
    switch_on_step=1000,
    projector_solver="auto",
    projector_maxiter=40,
    show_progress=True,
    momentum_projection_type="tangent_rand",
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
    ("loss", "Train loss"),
    ("accuracy", "Train batch accuracy"),
    ("chi_k_grad", "Gradient alignment with dom subspace"),
    ("chi_k_grad_ema", "EMA gradient alignment"),
    ("subspace_usefulness/rho", "Subspace usefulness ratio"),
    ("projector/subspace_pct", "Projection subspace size (% of parameters)"),
    ("validation/loss", "Validation loss"),
    ("validation/accuracy", "Validation accuracy"),
]


def write_experiment_config(out_dir: Path, device: str) -> None:
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_title": EXPERIMENT_TITLE,
        "device": device,
        "seeds": SEEDS,
        "run_plan_kwargs": RUN_PLAN_KWARGS,
        "runner_overrides": RUNNER_OVERRIDES,
        "plot_specs": PLOT_SPECS,
        "validation": {
            "dataset": "CIFAR10",
            "split": "test",
            "batch_size": 256,
            "normalization": {
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616],
            },
            "logged_every_training": True,
            "logged_on_runner_log_steps": True,
        },
    }
    with (out_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_device(arg: str = "auto") -> str:
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return arg


def run_label(run_name: str) -> str:
    if "__none__none" in run_name:
        return "baseline"
    if run_name.endswith("__dom"):
        return "dom"
    if run_name.endswith("__bulk"):
        return "bulk"
    return run_name


def make_cifar10_validation_loader(
    *,
    batch_size: int = 256,
    root: str = "./data",
    num_workers: int = 0,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_model_on_validation(
    *,
    model: torch.nn.Module,
    task,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        if task.batch_to_device is not None:
            batch = task.batch_to_device(batch, device, dtype)
        x, y = batch
        loss = task.loss_fn(model, batch)
        logits = model(x)
        batch_size = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == y).sum().detach().cpu())
        total_examples += batch_size

    if was_training:
        model.train()

    return {
        "validation/loss": total_loss / total_examples,
        "validation/accuracy": total_correct / total_examples,
    }


class ValidationLoggingRunner(ExperimentRunner):
    def __init__(
        self,
        *args,
        val_loader: DataLoader,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.val_loader = val_loader

    def _on_log_row(
        self,
        row: dict,
        *,
        step: int,
        loss_value: float,
        model: torch.nn.Module,
    ) -> None:
        del step, loss_value
        row.update(
            evaluate_model_on_validation(
                model=model,
                task=self.task,
                loader=self.val_loader,
                device=self.device,
                dtype=self.config.dtype,
            )
        )


def run_one_seed(
    seed: int,
    out_dir: Path,
    device: str,
    val_loader: DataLoader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    task, opts, projs, runner_cfg = build_run_plan(
        seed=seed,
        device=device,
        **RUN_PLAN_KWARGS,
    )
    runner_cfg.save_dir = seed_dir
    for key, value in RUNNER_OVERRIDES.items():
        setattr(runner_cfg, key, value)

    print(f"\nseed={seed} device={device}")
    print(f"optimizers={[o.name for o in opts]}")
    print(f"projectors={[p.name for p in projs]}")
    print(f"save_dir={seed_dir}")

    runner = ValidationLoggingRunner(task, opts, projs, runner_cfg, val_loader=val_loader)
    results = runner.run()

    history = results_to_dataframe(results)
    history["seed"] = seed
    history["run_label"] = history["run"].map(run_label)
    history.to_csv(seed_dir / "history.csv", index=False)

    validation_cols = [
        "seed",
        "run",
        "run_label",
        "step",
        "validation/loss",
        "validation/accuracy",
    ]
    validation = history[[c for c in validation_cols if c in history.columns]].copy()
    validation.to_csv(seed_dir / "validation_history.csv", index=False)

    return history, validation


def aggregate_history(history: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [
        metric
        for metric, _ in PLOT_SPECS
        if metric in history.columns and history[metric].notna().any()
    ]
    grouped = history.groupby(["run_label", "step"], as_index=False)[numeric_metrics]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)

    rows = []
    for _, mean_row in mean.iterrows():
        row = mean_row.to_dict()
        std_row = std.iloc[_]
        for metric in numeric_metrics:
            row[f"{metric}/std"] = std_row[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_validation(validation: pd.DataFrame) -> pd.DataFrame:
    if "step" in validation.columns:
        validation = validation.sort_values("step").groupby(
            ["seed", "run_label"],
            as_index=False,
        ).tail(1)
    metrics = ["validation/loss", "validation/accuracy"]
    mean = validation.groupby("run_label", as_index=False)[metrics].mean()
    std = validation.groupby("run_label", as_index=False)[metrics].std().fillna(0.0)
    rows = []
    for i, mean_row in mean.iterrows():
        row = mean_row.to_dict()
        for metric in metrics:
            row[f"{metric}/std"] = std.iloc[i][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_aggregate_dashboard(
    history_agg: pd.DataFrame,
    out_path: Path,
) -> None:
    available = [
        (metric, label)
        for metric, label in PLOT_SPECS
        if metric in history_agg.columns and history_agg[metric].notna().any()
    ]
    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 4.0 * nrows),
        squeeze=False,
        sharex=False,
    )
    axes_flat = axes.ravel()
    labels_seen: list[str] = []
    handles = []

    for ax, (metric, label) in zip(axes_flat, available):
        std_col = f"{metric}/std"
        for label_name, group in history_agg.groupby("run_label"):
            group = group.sort_values("step")
            x = group["step"].to_numpy()
            y = group[metric].to_numpy()
            line, = ax.plot(x, y, label=label_name)
            if label_name not in labels_seen:
                labels_seen.append(label_name)
                handles.append(line)
            if std_col in group.columns:
                std = group[std_col].fillna(0.0).to_numpy()
                ax.fill_between(x, y - std, y + std, alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    fig.suptitle(f"{EXPERIMENT_TITLE} ({len(SEEDS)} seeds)", fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels_seen,
            loc="lower center",
            ncol=min(len(labels_seen), 4),
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.05 if handles else 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.show()
def main() -> None:
    device = resolve_device("auto")
    out_dir = Path("local_runs") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_config(out_dir, device)
    val_loader = make_cifar10_validation_loader()

    all_histories = []
    all_validation = []
    for seed in SEEDS:
        history, validation = run_one_seed(seed, out_dir, device, val_loader)
        all_histories.append(history)
        all_validation.append(validation)

    history_all = pd.concat(all_histories, ignore_index=True)
    validation_all = pd.concat(all_validation, ignore_index=True)
    history_all.to_csv(out_dir / "history_all_seeds.csv", index=False)
    validation_all.to_csv(out_dir / "validation_all_seeds.csv", index=False)

    history_agg = aggregate_history(history_all)
    validation_agg = aggregate_validation(validation_all)
    history_agg.to_csv(out_dir / "history_aggregate.csv", index=False)
    validation_agg.to_csv(out_dir / "validation_aggregate.csv", index=False)

    print("\nValidation aggregate:")
    print(validation_agg.to_string(index=False))

    plot_aggregate_dashboard(
        history_agg=history_agg,
        out_path=out_dir / f"{EXPERIMENT_NAME}__dashboard.png",
    )
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
