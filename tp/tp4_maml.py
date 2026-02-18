"""
TP4 nano: MAML (5-way 1-shot Omniglot, linear 1LP) from
"Feature Learning in Infinite-Width Neural Networks" (arxiv 2011.14522).
Reference: https://github.com/edwardjhu/TP4 (TP4MAML)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Omniglot: 28x28, 5-way 1-shot
IN_FEATURES = 28 * 28


class SafeMetaLinear(nn.Linear):
    """Linear with optional bias scaling (bias_alpha) for meta-params."""

    def __init__(self, *args, bias_alpha: float = 1.0, **kwargs):
        self.bias_alpha = bias_alpha
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, params: dict | None = None) -> torch.Tensor:
        if params is not None:
            w = params["weight"]
            b = params.get("bias")
            if b is not None:
                b = b * self.bias_alpha
            return F.linear(input, w, b)
        return F.linear(
            input,
            self.weight,
            self.bias * self.bias_alpha if self.bias is not None else self.bias,
        )


class MetaFinLin1LP(nn.Module):
    """Finite-width linear 1-layer (1LP) for MAML; μP init so n→∞ matches MUP."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        width: int,
        alpha: float = 1.0,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
        bias_alpha1: float = 0.0,
        bias_alpha2: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.alpha = alpha
        have_bias1 = bias_alpha1 != 0
        have_bias2 = bias_alpha2 != 0
        self.features = SafeMetaLinear(
            in_features, width, bias=have_bias1, bias_alpha=bias_alpha1
        )
        self.classifier = SafeMetaLinear(
            width, out_features, bias=have_bias2, bias_alpha=bias_alpha2
        )
        with torch.no_grad():
            self.features.weight.normal_(0, sigma1 / math.sqrt(width))
            self.classifier.weight.zero_()
            if self.features.bias is not None:
                self.features.bias.zero_()
            if self.classifier.bias is not None:
                self.classifier.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        params: dict | None = None,
    ) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        if params is not None:
            feats = self.features(x, params=self._sub(params, "features"))
            logits = self.classifier(feats, params=self._sub(params, "classifier"))
        else:
            feats = self.features(x)
            logits = self.classifier(feats)
        return logits * self.alpha

    def _sub(self, params: dict, prefix: str) -> dict:
        return {k.replace(prefix + ".", ""): v for k, v in params.items() if k.startswith(prefix + ".")}

    def named_meta_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p


class MetaInfLin1LP(nn.Module):
    """Infinite-width MUP linear 1LP: identity-like first layer, learnable limit."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
        bias_alpha1: float = 0.0,
        bias_alpha2: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        h = in_features + out_features
        have_bias1 = bias_alpha1 != 0
        have_bias2 = bias_alpha2 != 0
        self.features = SafeMetaLinear(in_features, h, bias=have_bias1, bias_alpha=bias_alpha1)
        self.classifier = SafeMetaLinear(h, out_features, bias=have_bias2, bias_alpha=bias_alpha2)
        with torch.no_grad():
            self.features.weight[:in_features, :] = torch.eye(in_features) * sigma1
            self.features.weight[in_features:, :] = 0
            self.classifier.weight[:, :in_features] = 0
            self.classifier.weight[:, in_features:] = torch.eye(out_features) * sigma2
            if self.features.bias is not None:
                self.features.bias.zero_()
            if self.classifier.bias is not None:
                self.classifier.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        params: dict | None = None,
    ) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        if params is not None:
            feats = self.features(x, params=self._sub(params, "features"))
            logits = self.classifier(feats, params=self._sub(params, "classifier"))
        else:
            feats = self.features(x)
            logits = self.classifier(feats)
        return logits * self.alpha

    def _sub(self, params: dict, prefix: str) -> dict:
        return {k.replace(prefix + ".", ""): v for k, v in params.items() if k.startswith(prefix + ".")}

    def named_meta_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                yield n, p


def _gradient_update_parameters(
    model: nn.Module,
    loss: torch.Tensor,
    params: dict,
    step_size: float,
    first_order: bool,
) -> dict:
    param_list = [(n, p) for n, p in params.items() if p.requires_grad]
    grads = torch.autograd.grad(
        loss,
        [p for _, p in param_list],
        create_graph=not first_order,
        allow_unused=True,
    )
    g = {n: gr for (n, _), gr in zip(param_list, grads) if gr is not None}
    updated = {}
    for name, param in params.items():
        if name in g:
            updated[name] = param - step_size * g[name]
        else:
            updated[name] = param
    return updated


def _get_omniglot_tasks(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_ways: int,
    num_shots: int,
    num_query: int,
    num_tasks: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Sample episodic tasks: (train_x, train_y, test_x, test_y)."""
    unique_labels = labels.unique()
    tasks = []
    for _ in range(num_tasks):
        chosen = unique_labels[torch.randperm(len(unique_labels))[:num_ways]]
        train_x, train_y, test_x, test_y = [], [], [], []
        for way_idx, c in enumerate(chosen):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(idx))
            shot_idx = idx[perm[:num_shots]]
            query_idx = idx[perm[num_shots : num_shots + num_query]]
            for i in shot_idx:
                train_x.append(images[i])
                train_y.append(way_idx)
            for i in query_idx:
                test_x.append(images[i])
                test_y.append(way_idx)
        tasks.append(
            (
                torch.stack(train_x).to(device),
                torch.tensor(train_y, dtype=torch.long, device=device),
                torch.stack(test_x).to(device),
                torch.tensor(test_y, dtype=torch.long, device=device),
            )
        )
    return tasks


def _get_omniglot_data(root: str = "./data", download: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision.datasets import Omniglot
    except ImportError:
        raise ImportError("torchvision is required for MAML (Omniglot). Install with: pip install torchvision")
    import numpy as np
    train_ds = Omniglot(root=root, background=True, download=download)
    test_ds = Omniglot(root=root, background=False, download=download)

    def _to_tensor(img):
        arr = np.array(img)
        return torch.from_numpy(arr).float() / 255.0

    train_imgs = torch.stack([_to_tensor(train_ds[i][0]) for i in range(len(train_ds))])
    train_labs = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    test_imgs = torch.stack([_to_tensor(test_ds[i][0]) for i in range(len(test_ds))])
    test_labs = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    if train_imgs.dim() == 3:
        train_imgs = F.interpolate(
            train_imgs.unsqueeze(1), size=(28, 28), mode="bilinear", align_corners=False
        ).squeeze(1)
        test_imgs = F.interpolate(
            test_imgs.unsqueeze(1), size=(28, 28), mode="bilinear", align_corners=False
        ).squeeze(1)
    return train_imgs, train_labs, test_imgs, test_labs


def train_maml_nano(
    width: int | None = 32,
    inf_width: bool = False,
    num_epochs: int = 15,
    meta_lr: float = 0.1,
    inner_lr: float = 0.4,
    num_ways: int = 5,
    num_shots: int = 1,
    num_query: int = 1,
    num_inner_steps: int = 1,
    batch_size: int = 32,
    sigma1: float = 1.0,
    sigma2: float = 0.125,
    bias_alpha: float = 1.0,
    data_root: str = "./data",
    device: str | torch.device | None = None,
    seed: int = 42,
) -> tuple[nn.Module, list[float]]:
    """
    Train MAML (first-order) on Omniglot 5-way 1-shot with linear 1LP.
    Returns (model, accuracy_history per batch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    torch.manual_seed(seed)

    train_imgs, train_labs, _, _ = _get_omniglot_data(root=data_root)
    train_imgs = train_imgs.to(device)
    train_labs = train_labs.to(device)

    if inf_width:
        model = MetaInfLin1LP(
            IN_FEATURES,
            num_ways,
            alpha=1.0,
            sigma1=sigma1,
            sigma2=sigma2,
            bias_alpha1=bias_alpha,
            bias_alpha2=bias_alpha,
        ).to(device)
    else:
        model = MetaFinLin1LP(
            IN_FEATURES,
            num_ways,
            width=width or 32,
            sigma1=sigma1,
            sigma2=sigma2,
            bias_alpha1=bias_alpha,
            bias_alpha2=bias_alpha,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    acc_history: list[float] = []

    for epoch in range(num_epochs):
        tasks = _get_omniglot_tasks(
            train_imgs,
            train_labs,
            num_ways=num_ways,
            num_shots=num_shots,
            num_query=num_query,
            num_tasks=batch_size,
            device=device,
        )
        model.train()
        opt.zero_grad()
        total_acc = 0.0
        for (train_x, train_y, test_x, test_y) in tasks:
            params = dict(model.named_meta_parameters())
            for _ in range(num_inner_steps):
                logits = model(train_x, params=params)
                loss = F.cross_entropy(logits, train_y)
                params = _gradient_update_parameters(
                    model, loss, params, inner_lr, first_order=True
                )
            query_logits = model(test_x, params=params)
            outer_loss = F.cross_entropy(query_logits, test_y)
            outer_loss.backward()
            pred = query_logits.argmax(dim=1)
            total_acc += (pred == test_y).float().mean().item()
        opt.step()
        acc_history.append(total_acc / batch_size)

    return model, acc_history


def evaluate_maml_nano(
    model: nn.Module,
    num_ways: int = 5,
    num_shots: int = 1,
    num_query: int = 1,
    num_inner_steps: int = 5,
    inner_lr: float = 0.4,
    num_tasks: int = 200,
    data_root: str = "./data",
    device: str | torch.device | None = None,
    seed: int = 1,
) -> float:
    """Evaluate MAML model on Omniglot test split (background=False)."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    torch.manual_seed(seed)

    _, _, test_imgs, test_labs = _get_omniglot_data(root=data_root)
    test_imgs = test_imgs.to(device)
    test_labs = test_labs.to(device)

    tasks = _get_omniglot_tasks(
        test_imgs,
        test_labs,
        num_ways=num_ways,
        num_shots=num_shots,
        num_query=num_query,
        num_tasks=num_tasks,
        device=device,
    )
    model.eval()
    correct = 0
    total = 0
    for (train_x, train_y, test_x, test_y) in tasks:
        params = {
            n: p.detach().clone().requires_grad_(True)
            for n, p in model.named_meta_parameters()
        }
        for _ in range(num_inner_steps):
            logits = model(train_x, params=params)
            loss = F.cross_entropy(logits, train_y)
            grads = torch.autograd.grad(
                loss, list(params.values()), create_graph=False, allow_unused=True
            )
            params = {
                n: (p - inner_lr * g).detach().requires_grad_(True) if g is not None else p.detach().requires_grad_(True)
                for (n, p), g in zip(params.items(), grads)
            }
        with torch.no_grad():
            query_logits = model(test_x, params=params)
            pred = query_logits.argmax(dim=1)
            correct += (pred == test_y).sum().item()
            total += test_y.size(0)
    return correct / total if total else 0.0
