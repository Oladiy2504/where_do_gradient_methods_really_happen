from __future__ import annotations

import os
import re
from collections import Counter
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def get_mnist(batch_size: int = 50, root: str = "./data", num_workers: int = 2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    ds = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    ds = Subset(ds, range(5000))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_cifar10(batch_size: int = 50, root: str = "./data", num_workers: int = 2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    ds = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    ds = Subset(ds, range(5000))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


_WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


class WordLevelTokenizer:
    """Tiny word-level tokenizer built from a corpus.

    Normalization: lowercase, then regex split that keeps alphabetic words,
    numeric runs, and individual punctuation symbols as separate tokens.
    Special tokens: <pad>=0, <unk>=1.
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, texts: Iterable[str], min_freq: int = 1) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        self.token_to_id: dict[str, int] = {"<pad>": self.PAD_ID, "<unk>": self.UNK_ID}
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            self.token_to_id[token] = len(self.token_to_id)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.token_to_id.get(t, self.UNK_ID) for t in self._tokenize(text)]
        ids = ids[:max_len]
        attn = [1] * len(ids)
        pad_n = max_len - len(ids)
        if pad_n > 0:
            ids.extend([self.PAD_ID] * pad_n)
            attn.extend([0] * pad_n)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
        )


class SST2Dataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer: WordLevelTokenizer,
        max_len: int = 64,
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        input_ids, attention_mask = self.tokenizer.encode(item["sentence"], self.max_len)
        y = torch.tensor(item["label"], dtype=torch.long)
        return input_ids, attention_mask, y


def get_sst2(
    batch_size: int = 50,
    max_len: int = 64,
    num_workers: int = 0,
):
    raw = load_dataset("nyu-mll/glue", "sst2", split="train")
    raw = raw.select(range(1000))

    tokenizer = WordLevelTokenizer(raw["sentence"])

    ds = SST2Dataset(
        raw,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return loader, len(tokenizer)


# --------------------------- FineWeb (sp1024 LM) ---------------------------- #
#
# Pre-tokenized FineWeb shards from openai/parameter-golf, hosted on the HF
# dataset repo below. Tokens are uint16 ids over a 1024-entry SentencePiece
# vocab, so no tokenizer is needed at train time. Shards are 0-indexed; "first
# 8 shards" == indices 0..7. Each shard is ~200 MB (~100M tokens) and is
# memory-mapped, not loaded into RAM.

_FINEWEB_REPO = "willdepueoai/parameter-golf"
# The repo stores these under a doubled `datasets/datasets/...` prefix.
_FINEWEB_PREFIX = "datasets/datasets/fineweb10B_sp1024"
FINEWEB_VOCAB_SIZE = 1024
_FINEWEB_HEADER_INTS = 256
_FINEWEB_HEADER_BYTES = _FINEWEB_HEADER_INTS * np.dtype("<i4").itemsize  # 1024
_FINEWEB_MAGIC = 20240520


def _download_fineweb_shard(index: int, cache_dir: str) -> str:
    filename = f"{_FINEWEB_PREFIX}/fineweb_train_{index:06d}.bin"
    return hf_hub_download(
        repo_id=_FINEWEB_REPO,
        repo_type="dataset",
        filename=filename,
        cache_dir=cache_dir,
    )


def _memmap_fineweb_shard(path: str) -> np.memmap:
    header = np.fromfile(path, dtype="<i4", count=_FINEWEB_HEADER_INTS)
    if header.size != _FINEWEB_HEADER_INTS or int(header[0]) != _FINEWEB_MAGIC or int(header[1]) != 1:
        raise ValueError(f"Unexpected FineWeb shard header for {path}")
    num_tokens = int(header[2])
    return np.memmap(
        path, dtype="<u2", mode="r", offset=_FINEWEB_HEADER_BYTES, shape=(num_tokens,)
    )


class FineWebDataset(Dataset):
    """Contiguous next-token windows over one or more FineWeb token shards.

    Item i -> (input_ids, target_ids), both int64 of length ``seq_len``, where
    ``target_ids`` is ``input_ids`` shifted forward by one token. Windows never
    cross a shard boundary; tokens stay memory-mapped.
    """

    def __init__(self, shards: list[np.memmap], seq_len: int):
        self.shards = shards
        self.seq_len = seq_len
        counts = [max((len(s) - 1) // seq_len, 0) for s in shards]
        self._cum = np.cumsum([0] + counts)
        self.num_windows = int(self._cum[-1])
        if self.num_windows == 0:
            raise ValueError("FineWeb shards too short for the requested seq_len.")

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        s = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local = idx - int(self._cum[s])
        start = local * self.seq_len
        # asarray with a dtype change copies the memmap slice into a writable
        # int64 array, so torch.from_numpy is safe (no read-only warning).
        chunk = np.asarray(self.shards[s][start : start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def get_fineweb(
    seq_len: int = 256,
    train_batch_tokens: int = 16384,
    num_shards: int = 1,
    batch_size: int | None = None,
    cache_dir: str = "./data/fineweb",
    num_workers: int = 0,
):
    """FineWeb (parameter-golf sp1024, vocab=1024) next-token LM loader.

    Downloads the first ``num_shards`` training shards (0-indexed) on demand and
    memory-maps them. ``batch_size`` defaults to
    ``train_batch_tokens // seq_len`` (e.g. 16384 / 256 = 64 sequences/step),
    folding the token-budget batching into a plain sample batch size.

    Returns ``(loader, vocab_size)`` where the loader yields
    ``(input_ids, target_ids)`` of shape ``[batch_size, seq_len]`` — mirroring
    ``get_sst2``'s ``(loader, vocab_size)`` contract.
    """
    if batch_size is None:
        batch_size = train_batch_tokens // seq_len

    os.makedirs(cache_dir, exist_ok=True)
    shards = [
        _memmap_fineweb_shard(_download_fineweb_shard(i, cache_dir))
        for i in range(num_shards)
    ]

    ds = FineWebDataset(shards, seq_len=seq_len)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return loader, FINEWEB_VOCAB_SIZE
