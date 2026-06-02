"""
Тут собраны функции для сбора TaskSpec для раннера.

Тут описана предметную логика (datasets, models, loss, ...).

В основном цикле обучения отсюда нужны лишь make_..._task функции
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Literal, Mapping

import torch
import torch.nn.functional as F

from src.experiments.runner import Batch, MetricsFn, TaskSpec
from src.models.data import get_cifar10, get_fineweb, get_mnist, get_sst2
from src.models.cnn_cifar import CNN3CIFAR
from src.models.gpt import GPT
from src.models.mlp_mnist import MLP3MNIST
from src.models.resnet_cifar import ResNet8CIFAR
from src.models.transformer_sst import TransformerSST2
from src.models.vgg11_cifar import VGG11CIFAR

# forward_fn(model_like, batch) -> logits
ForwardFn = Callable[[Any, Batch], torch.Tensor]

# target_fn(batch) -> y
TargetFn = Callable[[Batch], torch.Tensor]


def batch_to_device(batch: Batch, device: torch.device, dtype: torch.dtype | None) -> Batch:
    """
    Move a nested batch to device. Floating tensors are optionally cast.
    """
    if torch.is_tensor(batch):
        if dtype is not None and batch.is_floating_point():
            return batch.to(device=device, dtype=dtype)
        return batch.to(device=device)

    if isinstance(batch, Mapping):
        return type(batch)({k: batch_to_device(v, device, dtype) for k, v in batch.items()})

    if isinstance(batch, tuple):
        return tuple(batch_to_device(x, device, dtype) for x in batch)

    if isinstance(batch, list):
        return [batch_to_device(x, device, dtype) for x in batch]

    return batch


def tuple_target(batch: Batch) -> torch.Tensor:
    return batch[-1]


def image_forward(model_like: Any, batch: Batch) -> torch.Tensor:
    """
    Общий forward для MNIST/CIFAR
    """
    x = batch[0]
    return model_like(x)


def sst2_forward(model_like: Any, batch: Batch) -> torch.Tensor:
    """
    Forward для SST2
    """
    input_ids, attention_mask, _ = batch
    return model_like(input_ids, attention_mask)


def lm_forward(model_like: Any, batch: Batch) -> torch.Tensor:
    """
    Forward для causal LM: возвращает logits [B, T, vocab].
    batch = (input_ids, target_ids).
    """
    return model_like(batch[0])


def lm_loss_fn(model_like: Any, batch: Batch) -> torch.Tensor:
    """
    Next-token cross-entropy для causal LM.
    """
    logits = lm_forward(model_like, batch)            # [B, T, V]
    targets = batch[1]                                # [B, T]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
    )


def make_lm_metric(forward_fn: ForwardFn = lm_forward) -> MetricsFn:
    """
    Token-accuracy + perplexity для causal LM.
    """
    def metrics_fn(model: torch.nn.Module, batch: Batch) -> dict[str, float]:
        logits = forward_fn(model, batch)             # [B, T, V]
        targets = batch[1]                            # [B, T]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
        )
        pred = logits.argmax(dim=-1)
        acc = (pred == targets).float().mean()
        return {
            "accuracy": float(acc.detach().cpu()),
            "perplexity": float(torch.exp(ce).detach().cpu()),
        }

    return metrics_fn


def make_classification_loss(forward_fn: ForwardFn, target_fn: TargetFn = tuple_target, 
                             loss_type: Literal["ce", "mse"] = "ce", num_classes: int | None = None) -> Callable[[Any, Batch], torch.Tensor]:
    """
    Factory для loss-функции

    Через loss_type можно задавать вид лосса
    """
    def loss_fn(model_like: Any, batch: Batch) -> torch.Tensor:
        logits = forward_fn(model_like, batch)
        y = target_fn(batch)

        if loss_type == "ce":
            return F.cross_entropy(logits.float(), y)

        if loss_type == "mse":
            if num_classes is None:
                raise ValueError("num_classes must be specified for categorical MSE.")
            # y_onehot = F.one_hot(y, num_classes=num_classes).float()
            # Изменил чтобы работал vmap
            classes = torch.arange(num_classes, device=y.device)
            y_onehot = (y.unsqueeze(-1) == classes).to(dtype=logits.dtype)
            # Sum over class dim, mean over batch (Hui & Belkin / Cohen et al.).
            return F.mse_loss(logits.float(), y_onehot, reduction="sum") / y.shape[0]

        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss_fn


def make_accuracy_metric(forward_fn: ForwardFn, target_fn: TargetFn = tuple_target) -> MetricsFn:
    """
    Factory для accuracy

    На инференсе:
    metrics = TaskSpec.metrics_fn(model, batch)
    """
    def metrics_fn(model: torch.nn.Module, batch: Batch) -> dict[str, float]:
        logits = forward_fn(model, batch)
        y = target_fn(batch)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        return {"accuracy": float(acc.detach().cpu())}

    return metrics_fn


def make_mnist_mlp3_task(batch_size: int = 50, root: str = "./data", num_workers: int = 0, input_dim: int = 28 * 28,
                         width: int = 200, num_classes: int = 10, loss_type: Literal["ce", "mse"] = "ce") -> TaskSpec:
    """
    Собирает TaskSpec для MNIST + MLP3MNIST

    model_factory - не фиксированная модель, а функция-фабрика для неё
    """
    loader = get_mnist(
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
    )

    model_factory = partial(
        MLP3MNIST,
        input_dim=input_dim,
        width=width,
        num_classes=num_classes,
    )

    return TaskSpec(
        name="mnist_mlp3",
        model_factory=model_factory,
        train_loader=loader,
        loss_fn=make_classification_loss(
            forward_fn=image_forward,
            target_fn=tuple_target,
            loss_type=loss_type,
            num_classes=num_classes if loss_type == "mse" else None,
        ),
        metrics_fn=make_accuracy_metric(
            forward_fn=image_forward,
            target_fn=tuple_target,
        ),
        batch_to_device=batch_to_device,
    )

def _make_cifar_image_task(name: str, model_factory: Callable[[], torch.nn.Module], loader, 
                           num_classes: int, loss_type: Literal["ce", "mse"]) -> TaskSpec:
    """
    Общий helper для сбора CIFAR-задач (чтобы не дублировать код)
    """
    return TaskSpec(
        name=name,
        model_factory=model_factory,
        train_loader=loader,
        loss_fn=make_classification_loss(
            forward_fn=image_forward,
            target_fn=tuple_target,
            loss_type=loss_type,
            num_classes=num_classes if loss_type == "mse" else None,
        ),
        metrics_fn=make_accuracy_metric(
            forward_fn=image_forward,
            target_fn=tuple_target,
        ),
        batch_to_device=batch_to_device,
    )


def make_cifar_cnn3_task(batch_size: int = 50, root: str = "./data", num_workers: int = 0,  width: int = 32,
                         num_classes: int = 10, loss_type: Literal["ce", "mse"] = "ce") -> TaskSpec:
    """
    Собирает CIFAR10 + CNN3CIFAR
    """
    loader = get_cifar10(
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
    )

    model_factory = partial(
        CNN3CIFAR,
        width=width,
        num_classes=num_classes,
    )

    return _make_cifar_image_task(
        name="cifar10_cnn3",
        model_factory=model_factory,
        loader=loader,
        num_classes=num_classes,
        loss_type=loss_type,
    )


def make_cifar_resnet8_task(batch_size: int = 50, root: str = "./data", num_workers: int = 0, 
                            base_width: int = 16, num_classes: int = 10, loss_type: Literal["ce", "mse"] = "ce") -> TaskSpec:
    """
    Собирает CIFAR10 + ResNet8CIFAR
    """
    loader = get_cifar10(
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
    )

    model_factory = partial(
        ResNet8CIFAR,
        num_classes=num_classes,
        base_width=base_width,
    )

    return _make_cifar_image_task(
        name="cifar10_resnet8",
        model_factory=model_factory,
        loader=loader,
        num_classes=num_classes,
        loss_type=loss_type,
    )


def make_cifar_vgg11_task(batch_size: int = 50, root: str = "./data", num_workers: int = 0, num_classes: int = 10,
                          dropout: bool = False, batch_norm: bool = False, loss_type: Literal["ce", "mse"] = "ce") -> TaskSpec:
    """
    Собирает CIFAR10 + VGG11CIFAR
    """
    loader = get_cifar10(
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
    )

    model_factory = partial(
        VGG11CIFAR,
        num_classes=num_classes,
        dropout=dropout,
        batch_norm=batch_norm,
    )

    return _make_cifar_image_task(
        name="cifar10_vgg11",
        model_factory=model_factory,
        loader=loader,
        num_classes=num_classes,
        loss_type=loss_type,
    )


def make_sst2_transformer_task(batch_size: int = 50, max_len: int = 64, num_workers: int = 0,
                               hidden_dim: int = 64, num_heads: int = 8, num_layers: int = 2, num_classes: int = 2,
                               dropout: float = 0.0, loss_type: Literal["ce", "mse"] = "ce") -> TaskSpec:
    """
    Собирает SST2 + TransformerSST2
    """
    loader, vocab_size = get_sst2(
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
    )

    model_factory = partial(
        TransformerSST2,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        num_classes=num_classes,
        dropout=dropout,
    )

    return TaskSpec(
        name="sst2_transformer",
        model_factory=model_factory,
        train_loader=loader,
        loss_fn=make_classification_loss(
            forward_fn=sst2_forward,
            target_fn=tuple_target,
            loss_type=loss_type,
            num_classes=num_classes if loss_type == "mse" else None,
        ),
        metrics_fn=make_accuracy_metric(
            forward_fn=sst2_forward,
            target_fn=tuple_target,
        ),
        batch_to_device=batch_to_device,
    )


def make_fineweb_gpt_task(
    seq_len: int = 256,
    train_batch_tokens: int = 16384,
    num_shards: int = 1,
    num_workers: int = 0,
    vocab_size: int = 1024,
    num_layers: int = 2,
    model_dim: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    mlp_mult: int = 2,
) -> TaskSpec:
    """
    Собирает FineWeb (sp1024) + GPT-бейзлайн (openai/parameter-golf).

    Causal LM: loss_fn = next-token cross-entropy. Токен-бюджет
    train_batch_tokens сворачивается в batch_size = train_batch_tokens // seq_len
    внутри get_fineweb. num_shards=1 по умолчанию (до 8 — просто увеличить).
    """
    loader, vocab = get_fineweb(
        seq_len=seq_len,
        train_batch_tokens=train_batch_tokens,
        num_shards=num_shards,
        num_workers=num_workers,
    )

    model_factory = partial(
        GPT,
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
    )

    return TaskSpec(
        name="fineweb_gpt",
        model_factory=model_factory,
        train_loader=loader,
        loss_fn=lm_loss_fn,
        metrics_fn=make_lm_metric(),
        batch_to_device=batch_to_device,
    )
