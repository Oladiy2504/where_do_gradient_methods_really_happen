from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)


Closure = Callable[[], torch.Tensor]


class SWA(Optimizer):
    update_bn_on_finalize = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        swa_start: int = 0,
        swa_freq: int = 1,
        lr_min: float | None = None,
        cycle_length: int | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if swa_start < 0:
            raise ValueError(f"Invalid swa_start: {swa_start}")
        if swa_freq <= 0:
            raise ValueError(f"Invalid swa_freq: {swa_freq}")
        if lr_min is not None and lr_min < 0.0:
            raise ValueError(f"Invalid lr_min: {lr_min}")
        if cycle_length is not None and cycle_length <= 0:
            raise ValueError(f"Invalid cycle_length: {cycle_length}")
        if (lr_min is None) != (cycle_length is None):
            raise ValueError("lr_min and cycle_length must be set together.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
            swa_start=int(swa_start),
            swa_freq=int(swa_freq),
            lr_min=lr_min,
            cycle_length=None if cycle_length is None else int(cycle_length),
        )
        super().__init__(params, defaults)
        self.last_info: dict[str, object] = {}

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Closure] = None,
        *,
        projector: BaseProjector | None = None,
        projection: ProjectionMode = "none",
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if projection != "none" and projector is None:
            raise ValueError("projection is not 'none', but projector is None.")

        all_params = [p for group in self.param_groups for p in group["params"]]
        global_state = self._global_state(all_params)

        current_step = int(global_state["step"]) + 1
        raw_update: list[torch.Tensor] = []
        step_lrs: list[float] = []

        if int(global_state["n_averaged"]) == 0 and self._min_swa_start() == 0:
            self._initialize_swa_average(all_params, global_state)

        for group in self.param_groups:
            lr = self._scheduled_lr(group, current_step)
            step_lrs.append(lr)
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p))
                    continue

                grad = p.grad.float()
                if wd != 0.0:
                    grad = grad.add(p.float(), alpha=wd)

                state = self.state[p]
                if momentum != 0.0:
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = (
                            torch.clone(grad).detach().float()
                        )
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    direction = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    direction = grad

                raw_update.append(lr * direction)

        update, info = project_or_passthrough(tuple(raw_update), projector, projection)

        for p, u in zip(all_params, update):
            if p.grad is None:
                continue
            p.add_(u.to(dtype=p.dtype), alpha=-1.0)

        global_state["step"] = current_step
        swa_updated = self._maybe_update_swa_average(
            all_params,
            global_state,
            current_step,
        )

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals,
            "swa_step": current_step,
            "swa_n_averaged": int(global_state["n_averaged"]),
            "swa_updated": swa_updated,
            "swa_lr": step_lrs[0] if len(step_lrs) == 1 else step_lrs
        }

        return loss

    @torch.no_grad()
    def apply_swa_weights(self) -> bool:
        """Копирует накопленное среднее в живые параметры модели."""
        all_params = [p for group in self.param_groups for p in group["params"]]
        if not all_params:
            return False

        global_state = self._global_state(all_params)
        if int(global_state["n_averaged"]) == 0:
            return False

        for p in all_params:
            buf = self.state[p].get("swa_buffer")
            if buf is not None:
                p.copy_(buf.to(device=p.device, dtype=p.dtype))
        self.last_info = {
            **self.last_info,
            "swa_applied": True,
            "swa_n_averaged": int(global_state["n_averaged"])
        }
        return True

    def _global_state(self, params: list[torch.nn.Parameter]) -> dict[str, object]:
        if not params:
            raise ValueError("SWA got an empty parameter list.")
        state = self.state[params[0]]
        state.setdefault("step", 0)
        state.setdefault("n_averaged", 0)
        return state

    def _min_swa_start(self) -> int:
        return min(int(group["swa_start"]) for group in self.param_groups)

    def _initialize_swa_average(
        self,
        params: list[torch.nn.Parameter],
        global_state: dict[str, object],
    ) -> None:
        for p in params:
            self.state[p]["swa_buffer"] = p.detach().clone().float()
        global_state["n_averaged"] = 1

    def _maybe_update_swa_average(
        self,
        params: list[torch.nn.Parameter],
        global_state: dict[str, object],
        current_step: int,
    ) -> bool:
        start = self._min_swa_start()
        freq = min(int(group["swa_freq"]) for group in self.param_groups)
        if current_step < start:
            return False
        if (current_step - start) % freq != 0:
            return False

        n_averaged = int(global_state["n_averaged"])
        if n_averaged == 0:
            self._initialize_swa_average(params, global_state)
            return True

        new_count = n_averaged + 1
        for p in params:
            state = self.state[p]
            if "swa_buffer" not in state:
                state["swa_buffer"] = p.detach().clone().float()
            else:
                state["swa_buffer"].add_(
                    p.detach().float() - state["swa_buffer"],
                    alpha=1.0 / new_count,
                )
        global_state["n_averaged"] = new_count
        return True

    @staticmethod
    def _scheduled_lr(group: dict[str, object], step: int) -> float:
        lr = float(group["lr"])
        lr_min = group.get("lr_min")
        cycle_length = group.get("cycle_length")
        if lr_min is None or cycle_length is None:
            return lr

        c = int(cycle_length)
        t = ((step - 1) % c + 1) / c
        return (1.0 - t) * lr + t * float(lr_min)
