from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from datasets import load_dataset


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
    raw = load_dataset("glue", "sst2", split="train")
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
