"""
TP4 nano: Word2Vec (CBOW + negative sampling) from
"Feature Learning in Infinite-Width Neural Networks" (arxiv 2011.14522).
Reference: https://github.com/edwardjhu/TP4 (Word2Vec)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_vocab(
    corpus: list[str],
    min_count: int = 2,
    max_vocab: int = 5000,
) -> tuple[list[str], dict[str, int]]:
    """Build vocabulary from tokenized corpus."""
    counts = Counter(corpus)
    vocab = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
    vocab = vocab[:max_vocab]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.lower().split()


def _get_cbow_pairs(
    tokens: list[str],
    word2idx: dict[str, int],
    window: int = 2,
) -> list[tuple[list[int], int]]:
    """Yield (context_indices, center_index) for CBOW."""
    pairs = []
    for i in range(window, len(tokens) - window):
        center = tokens[i]
        context = (
            tokens[i - window : i] + tokens[i + 1 : i + 1 + window]
        )
        if center in word2idx and all(c in word2idx for c in context):
            context_idx = [word2idx[tokens[j]] for j in range(i - window, i)] + [
                word2idx[tokens[j]] for j in range(i + 1, i + 1 + window)
            ]
            center_idx = word2idx[center]
            pairs.append((context_idx, center_idx))
    return pairs


def _negative_sample(
    batch_size: int,
    num_neg: int,
    vocab_size: int,
    positive_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample negative indices (excluding positive)."""
    neg = torch.randint(0, vocab_size, (batch_size, num_neg), device=device)
    for b in range(batch_size):
        pos = positive_idx[b].item()
        for k in range(num_neg):
            while neg[b, k].item() == pos:
                neg[b, k] = torch.randint(0, vocab_size, (1,), device=device).item()
    return neg


class CBOWFinite(nn.Module):
    """CBOW with one hidden layer; μP scaling (1/sqrt(width))."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        width: int,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.width = width
        scale1 = sigma1 / math.sqrt(width)
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, width)
        self.fc2 = nn.Linear(width, vocab_size)
        nn.init.normal_(self.fc1.weight, 0, scale1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.normal_(self.in_embed.weight, 0, 0.1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        emb = self.in_embed(context).mean(dim=1)
        h = self.fc1(emb) / math.sqrt(self.width)
        logits = self.fc2(h) / math.sqrt(self.width)
        return logits


class CBOWInf(nn.Module):
    """Infinite-width μP limit (fixed large representation as limit proxy for nano)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        limit_dim: int = 2048,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.limit_dim = limit_dim
        scale1 = sigma1 / math.sqrt(limit_dim)
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, limit_dim)
        self.fc2 = nn.Linear(limit_dim, vocab_size)
        nn.init.normal_(self.fc1.weight, 0, scale1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.normal_(self.in_embed.weight, 0, 0.1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        emb = self.in_embed(context).mean(dim=1)
        h = self.fc1(emb) / math.sqrt(self.limit_dim)
        logits = self.fc2(h) / math.sqrt(self.limit_dim)
        return logits


def _prepare_corpus(
    corpus_or_path: str | list[str],
    max_tokens: int = 50_000,
    min_count: int = 2,
    max_vocab: int = 3000,
) -> tuple[list[str], list[str], dict[str, int], int]:
    """Return (token_strings, vocab_list, word2idx, window)."""
    if isinstance(corpus_or_path, str):
        with open(corpus_or_path, "r") as f:
            text = f.read()
        tokens = _tokenize(text)[:max_tokens]
    else:
        tokens = list(corpus_or_path)[:max_tokens]
    vocab, word2idx = _build_vocab(tokens, min_count=min_count, max_vocab=max_vocab)
    token_strings = [w for w in tokens if w in word2idx]
    window = 2
    return token_strings, vocab, word2idx, window


def train_word2vec_nano(
    corpus_or_path: str | list[str],
    width: int | None = 64,
    inf_width: bool = False,
    epochs: int = 5,
    lr: float = 0.05,
    wd: float = 0.001,
    batch_size: int = 128,
    num_neg: int = 5,
    embed_dim: int = 64,
    max_tokens: int = 30_000,
    max_vocab: int = 2000,
    device: str | torch.device | None = None,
    seed: int = 42,
) -> tuple[Any, list[float]]:
    """
    Train nano Word2Vec (CBOW + negative sampling) with μP scaling.
    Returns (model, loss_history).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    torch.manual_seed(seed)
    random.seed(seed)

    token_strings, vocab, word2idx, window = _prepare_corpus(
        corpus_or_path, max_tokens=max_tokens, max_vocab=max_vocab
    )
    vocab_size = len(vocab)
    pairs = _get_cbow_pairs(token_strings, word2idx, window=window)
    if not pairs:
        raise ValueError("No CBOW pairs found; corpus or vocab too small.")

    if inf_width:
        model = CBOWInf(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            limit_dim=2048,
        ).to(device)
    else:
        model = CBOWFinite(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            width=width or 64,
        ).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    loss_history: list[float] = []

    for epoch in range(epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            if len(batch_pairs) < 2:
                continue
            context_list = [p[0] for p in batch_pairs]
            center_list = [p[1] for p in batch_pairs]
            ctx = torch.tensor(context_list, dtype=torch.long, device=device)
            pos = torch.tensor(center_list, dtype=torch.long, device=device)
            neg = _negative_sample(
                len(batch_pairs), num_neg, vocab_size, pos, device
            )
            logits = model(ctx)
            pos_logits = logits.gather(1, pos.unsqueeze(1)).squeeze(1)
            neg_logits = logits.gather(1, neg)
            loss = -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches:
            loss_history.append(epoch_loss / n_batches)

    return model, loss_history
