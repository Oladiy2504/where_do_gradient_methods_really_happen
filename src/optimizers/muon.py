from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Literal

import torch
from torch.optim.optimizer import Optimizer

from src.projections import (
    BaseProjector,
    ProjectionMode,
    project_or_passthrough,
)


Closure = Callable[[], torch.Tensor]
PolynomialMode = Literal["jordan", "cans", "polarexpress"]


_CLASSIC5_COEFFS: tuple[float, float, float] = (1.875, -1.25, 0.375)

_POLAR_EXPRESS5_RAW_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)

_POLAR_EXPRESS5_COEFFS: tuple[tuple[float, float, float], ...] = tuple(
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _POLAR_EXPRESS5_RAW_COEFFS[:-1]
) + (_POLAR_EXPRESS5_RAW_COEFFS[-1],)


def _cans_explicit3(a: float, b: float) -> tuple[float, float, float]:
    """Optimal odd cubic CANS polynomial on [a, b].

    Returns coefficients c1, c3 and approximation error err for
        p(x) = c1 * x + c3 * x^3.
    """
    s = a * a + a * b + b * b
    e = math.sqrt(s / 3.0)
    denom = 2.0 * e**3 + a * a * b + b * b * a

    c1 = 2.0 * s / denom
    c3 = -2.0 / denom
    err = (2.0 * e**3 - a * a * b - b * b * a) / denom
    return c1, c3, err

def _delta_orthogonalization(n=1, delta=0.3, B=1):
    """Find polynomial coefficients for the delta-orthogonalization phase.

    This is the preprocessing phase for CANS.

    Uses binary search to find the left endpoint Al such that n preprocessing
    iterations of explicit3 polynomials map singular values from [Al, B]
    into [1 - delta, 1 + delta].

    Returns the list of (c1, c3, err) coefficients for each iteration and
    the found left endpoint Al.
    """
    Al = 0.0
    Ar = B
    e = 100
    
    coeffs_list: list[tuple[float, float, float]] = []
    
    while abs(e - delta) > 1e-7:
        a, b = (Al + Ar) / 2.0, B
        coeffs_list = []
        for _ in range(n):
            c1, c3, e = _cans_explicit3(a, b)
            coeffs_list.append((c1, c3, e))
            a, b = 1.0 - e, 1.0 + e
        if e < delta:
            Ar = (Al + Ar) / 2.0
        else:
            Al = (Al + Ar) / 2.0
    
    return coeffs_list, (Al + Ar) / 2.0
        

def _muon_orthogonalize(
    x: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    full_ort: bool = False,
    polynom: PolynomialMode = "jordan",
    cans_a: float = 1e-8,
    cans_preprocess_steps: int = 0,
    cans_delta: float = 0.99,
) -> torch.Tensor:
    """Newton-Schulz orthogonalisation for tensors with ``ndim >= 2``.

    Higher-rank tensors (e.g. conv weights with ``ndim == 4``) are flattened
    to ``[fan_out, -1]`` for the iteration and reshaped back afterwards, so
    they actually receive the orthogonalisation step rather than passing
    through unchanged.
    """
    if x.ndim < 2:
        return x
    if polynom not in ("jordan", "cans", "polarexpress"):
        raise ValueError(
            f"Unknown polynom={polynom!r}. Expected 'jordan', 'cans' or 'polarexpress'."
        )

    original_shape = x.shape
    original_dtype = x.dtype

    if x.ndim > 2:
        m = x.reshape(x.shape[0], -1)
    else:
        m = x

    y = m.float()

    if full_ort:
        u, _, vh = torch.linalg.svd(y, full_matrices=False)
        y = u @ vh
        return y.to(dtype=original_dtype).reshape(original_shape)

    transposed = False
    if y.shape[0] > y.shape[1]:
        y = y.t()
        transposed = True

    if polynom == "jordan":
        a, b, c = 3.4445, -4.7750, 2.0315
        y = y / (y.norm() + eps)

        for _ in range(steps):
            yy = y @ y.t()
            y = a * y + (b * yy + c * (yy @ yy)) @ y
    elif polynom == "cans":
        one_norm = torch.linalg.norm(y, ord=1)
        inf_norm = torch.linalg.norm(y, ord=float("inf"))
        scale = torch.rsqrt((one_norm * inf_norm).clamp_min(eps))
        y = y * scale

        if cans_preprocess_steps > 0:
            pre_coeffs, _ = _delta_orthogonalization(
                n=cans_preprocess_steps, delta=cans_delta
            )
            for c1, c3, _ in pre_coeffs:
                yy = y @ y.t()
                y = c1 * y + c3 * (yy @ y)
            
            left, right = 1.0 - cans_delta, 1.0 + cans_delta
        else:
            left, right = cans_a, 1.0
            
        for _ in range(steps):
            c1, c3, err = _cans_explicit3(left, right)
            yy = y @ y.t()
            y = c1 * y + c3 * (yy @ y)

            left, right = 1.0 - err, 1.0 + err



    else:
        y = y / (y.norm() * 1.01 + eps)
        for i in range(steps):
            c1, c3, c5 = (
                _POLAR_EXPRESS5_COEFFS[i]
                if i < len(_POLAR_EXPRESS5_COEFFS)
                else _CLASSIC5_COEFFS
            )
            yy = y @ y.t()
            y = c1 * y + (c3 * yy + c5 * (yy @ yy)) @ y

    if transposed:
        y = y.t()

    return y.to(dtype=original_dtype).reshape(original_shape)



class Muon(Optimizer):
    """Heavy-ball SGD + Newton-Schulz orthogonalisation.

    The ``orth_after_projection`` flag controls the order of orthogonalisation
    and subspace projection:

    * ``True`` (default, backward-compatible): project the momentum direction
      first, then orthogonalise the projected vector. The applied update
      generally leaves the subspace ``Q`` again -- ``last_info["alignment"]``
      describes the momentum alignment, not the alignment of the actually
      applied update.
    * ``False``: orthogonalise the momentum direction first, then project.
      The applied update truly lives in ``Q`` (for ``dom``) or its complement
      (for ``bulk``) and ``last_info["alignment"]`` faithfully describes it.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        full_ort: bool = False,
        orth_after_projection: bool = True,
        polynom: PolynomialMode = "jordan",
        cans_a: float = 1e-8,
        cans_preprocess_steps: int = 4,
        cans_delta: float = 0.3,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if polynom not in ("jordan", "cans", "polarexpress"):
            raise ValueError(
                f"Unknown polynom={polynom!r}. Expected 'jordan', 'cans' or 'polarexpress'."
            )
        if cans_a <= 0.0:
            raise ValueError(f"Invalid cans_a: {cans_a}")
        if cans_preprocess_steps < 0:
            raise ValueError(f"Invalid cans_preprocess_steps: {cans_preprocess_steps}")
        if not (0.0 < cans_delta < 1.0):
            raise ValueError(f"Invalid cans_delta: {cans_delta}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            full_ort=full_ort,
            orth_after_projection=orth_after_projection,
            polynom=polynom,
            cans_a=cans_a,
            cans_preprocess_steps=cans_preprocess_steps,
            cans_delta=cans_delta,
        )
        super().__init__(params, defaults)
        self.last_info: dict[str, object] = {}

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
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
        raw_update: list[torch.Tensor] = []

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    raw_update.append(torch.zeros_like(p, dtype=torch.float32))
                    continue

                grad = p.grad.float()

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                direction = grad.add(buf, alpha=momentum) if nesterov else buf

                raw_update.append(direction)

        # Per-parameter settings indexed in the flat order used above.
        per_param_settings: list[tuple[int, bool, bool, PolynomialMode, float]] = []
        for group in self.param_groups:
            for _ in group["params"]:
                per_param_settings.append(
                    (
                        group["ns_steps"],
                        group["full_ort"],
                        group["orth_after_projection"],
                        group["polynom"],
                        group["cans_a"],
                        group["cans_preprocess_steps"],
                        group["cans_delta"],
                    )
                )

        if any(not s[2] for s in per_param_settings):
            # Orthogonalise first, then project. The applied update lives in Q.
            pre_proj: list[torch.Tensor] = []
            for p, u, (ns_steps, full_ort, _, polynom, cans_a, cans_preprocess_steps, cans_delta) in zip(
                all_params, raw_update, per_param_settings
            ):
                if p.grad is not None and p.ndim >= 2:
                    pre_proj.append(
                        _muon_orthogonalize(
                            u,
                            steps=ns_steps,
                            full_ort=full_ort,
                            polynom=polynom,
                            cans_a=cans_a,
                            cans_preprocess_steps=cans_preprocess_steps,
                            cans_delta=cans_delta,
                        )
                    )
                else:
                    pre_proj.append(u)
            update, info = project_or_passthrough(
                tuple(pre_proj), projector, projection
            )
        else:
            update, info = project_or_passthrough(
                tuple(raw_update), projector, projection
            )

        self.last_info = {
            "projection": projection if projector is not None else "none",
            "raw_update_norm": info.raw_norm,
            "projected_update_norm": info.projected_norm,
            "alignment": info.alignment,
            "eigvals": info.eigvals,
            "polynom": self.param_groups[0]["polynom"],
        }

        idx = 0

        for group in self.param_groups:
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            full_ort = group["full_ort"]
            orth_after_projection = group["orth_after_projection"]
            polynom = group["polynom"]
            cans_a = group["cans_a"]
            cans_preprocess_steps = group["cans_preprocess_steps"]
            cans_delta = group["cans_delta"]

            for p in group["params"]:
                u = update[idx]
                idx += 1

                if p.grad is None:
                    continue

                if orth_after_projection and p.ndim >= 2:
                    u = _muon_orthogonalize(
                        u,
                        steps=ns_steps,
                        full_ort=full_ort,
                        polynom=polynom,
                        cans_a=cans_a,
                        cans_preprocess_steps=cans_preprocess_steps,
                        cans_delta=cans_delta,
                    )

                # Decoupled weight decay sits outside the projection.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(u.to(dtype=p.dtype), alpha=-lr)

        return loss 