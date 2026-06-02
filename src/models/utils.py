import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable


OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS requires a workspace config for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def categorical_mse_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    # Sum over class dim, mean over batch (Hui & Belkin / Cohen et al.).
    return F.mse_loss(logits, y_onehot, reduction="sum") / y.shape[0]


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_image_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    steps: int,
    num_classes: int,
    optimizer_factory: OptimizerFactory,
    loss_type: str = "mse",
    log_every: int = 100,
):
    model.to(device=device, dtype=torch.bfloat16)
    model.train()

    optimizer = optimizer_factory(model)

    data_iter = cycle(loader)

    for step in range(1, steps + 1):
        x, y = next(data_iter)
        x = x.to(device=device, dtype=torch.bfloat16)
        y = y.to(device)

        logits = model(x)

        if loss_type == "mse":
            loss = categorical_mse_loss(logits, y, num_classes)
        elif loss_type == "ce":
            loss = F.cross_entropy(logits.float(), y)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            acc = accuracy(logits.detach(), y)
            print(f"step={step:06d} loss={loss.item():.6f} acc={acc:.4f}")

    return model


def train_text_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    steps: int,
    optimizer_factory: OptimizerFactory,
    num_classes: int = 2,
    loss_type: str = "mse",
    log_every: int = 100,
):
    model.to(device=device, dtype=torch.bfloat16)
    model.train()

    optimizer = optimizer_factory(model)

    data_iter = cycle(loader)

    for step in range(1, steps + 1):
        input_ids, attention_mask, y = next(data_iter)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        logits = model(input_ids, attention_mask)

        if loss_type == "mse":
            loss = categorical_mse_loss(logits, y, num_classes)
        elif loss_type == "ce":
            loss = F.cross_entropy(logits.float(), y)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            acc = accuracy(logits.detach(), y)
            print(f"step={step:06d} loss={loss.item():.6f} acc={acc:.4f}")

    return model
