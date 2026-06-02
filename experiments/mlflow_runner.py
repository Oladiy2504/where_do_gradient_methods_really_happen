"""MLflow-aware subclass of ExperimentRunner.

Each (task, optimizer, projector, projection) tuple is mapped to one MLflow
run inside the user-provided experiment. Params are logged once at run start;
metrics are streamed at every log-step from `_make_log_row`; the model
state_dict with the lowest train loss seen so far is uploaded as an artifact
under `best_ckpt/`.

This subclass intentionally owns no training-loop code: it plugs into the
canonical loop in `ExperimentRunner._run_one` via the observer hooks
(`_on_run_start`, `_on_log_row`, `_on_run_finished`, `_on_run_failed`).
Future fixes to the training loop (e.g. full-batch chi_k from Song et al.
Section 3.2) therefore reach MLflow runs automatically.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any, Iterator

import mlflow
import torch

from src.experiments.runner import (
    ExperimentRunner,
    OptimizerSpec,
    ProjectionMode,
    ProjectorSpec,
    SwitchCheckpoint,
)


_METRIC_KEYS = (
    "loss",
    "accuracy",
    "chi_k",
    "chi_k_ema",
    "update/raw_update_norm",
    "update/projected_update_norm",
    "update/alignment",
    "subspace_usefulness/rho",
    "epoch_time_sec",
    "epoch_time_sec_avg",
)


def _to_param_value(v: Any) -> str:
    """MLflow params must be primitive-ish. We stringify dicts/lists."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v if isinstance(v, str) else str(v)
    return str(v)


class MLflowLoggingRunner(ExperimentRunner):
    """Same as ExperimentRunner, but streams everything into MLflow.

    Overrides the four observer hooks added to `ExperimentRunner` so the
    training loop itself is untouched. Per-run state (`_best_loss`,
    `_best_path`, `_tmp_dir`) is initialised in `_on_run_start` because one
    runner instance executes the full plan and each `_run_one` call needs
    a fresh checkpoint scratch space.
    """

    @contextlib.contextmanager
    def _on_run_start(
            self,
            *,
            run_name: str,
            opt_spec: OptimizerSpec,
            proj_spec: ProjectorSpec | None,
            projection: ProjectionMode,
            seed: int,
            resume_from: SwitchCheckpoint | None = None,
    ) -> Iterator[None]:
        projector_name = "none" if proj_spec is None else proj_spec.name
        params_to_log: dict[str, Any] = {
            "task": self.task.name,
            "optimizer": opt_spec.name,
            "optimizer_kind": opt_spec.kind,
            "optimizer_kwargs": opt_spec.kwargs,
            "projector": projector_name,
            "projection": projection,
            "steps": self.config.steps,
            "device": str(self.device),
            "dtype": str(self.config.dtype),
            "seed": seed,
            "chi_ema_factor": self.config.chi_ema_factor,
        }
        if proj_spec is not None:
            params_to_log.update(
                {
                    "projector_kwargs": proj_spec.kwargs,
                    "projector_modes": list(proj_spec.modes),
                    "update_kind": proj_spec.update_kind,
                    "update_every_steps": proj_spec.update_every_steps,
                    "basis_full_dataset": proj_spec.basis_full_dataset,
                    "switch_on_alignment_ema": proj_spec.switch_on_alignment_ema,
                }
            )
        if resume_from is not None:
            params_to_log.update(
                {
                    "resumed_from_switch": True,
                    "resume_from_step": resume_from.step,
                    "resume_from_chi_ema": resume_from.chi_ema,
                    "paired_dom_run_id": resume_from.paired_dom_run_id,
                }
            )

        with mlflow.start_run(run_name=run_name) as mlrun:
            for k, v in params_to_log.items():
                mlflow.log_param(k, _to_param_value(v))

            self._best_loss: float = float("inf")
            self._best_path: Path | None = None
            self._tmp_dir: Path = Path(
                tempfile.mkdtemp(prefix=f"mlflow_best_{mlrun.info.run_id}_")
            )
            yield

    def _capture_switch_checkpoint(self, **kwargs: Any) -> SwitchCheckpoint:
        ckpt = super()._capture_switch_checkpoint(**kwargs)
        active = mlflow.active_run()
        if active is not None:
            ckpt.paired_dom_run_id = active.info.run_id
        return ckpt

    def _on_log_row(
            self,
            row: dict[str, Any],
            *,
            step: int,
            loss_value: float,
            model: torch.nn.Module,
    ) -> None:
        metrics: dict[str, float] = {}
        for key in _METRIC_KEYS:
            val = row.get(key)
            if isinstance(val, (int, float)) and val is not None:
                metrics[key.replace("/", "_")] = float(val)
        if metrics:
            mlflow.log_metrics(metrics, step=step)

        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_path = self._tmp_dir / "best_ckpt.pt"
            torch.save(
                {"step": step, "loss": loss_value, "state_dict": model.state_dict()},
                self._best_path,
            )

    def _on_run_finished(
            self,
            *,
            run_name: str,
            history: list[dict[str, Any]],
            model: torch.nn.Module,
    ) -> None:
        del run_name, model
        if self._best_path is not None and self._best_path.exists():
            mlflow.log_artifact(str(self._best_path), artifact_path="best_ckpt")
        if history:
            final_loss = history[-1].get("loss", float("nan"))
            mlflow.log_metric("final_loss", final_loss)
            switched = history[-1].get("switched_at_step")
            if switched is not None:
                mlflow.log_metric("switched_at_step", switched)

    def _on_run_failed(self, run_name: str, error: str) -> None:
        del run_name
        mlflow.log_param("status", "failed")
        mlflow.log_text(error, "error.txt")
