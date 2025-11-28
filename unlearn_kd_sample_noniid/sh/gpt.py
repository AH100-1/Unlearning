# 📦 Project: unlearn_kd — CIFAR‑10 Sample Unlearning + Knowledge Distillation

# ─────────────────────────────────────────────────────────────────────────────
# Directory Layout (create exactly this)
# ─────────────────────────────────────────────────────────────────────────────
# unlearn_kd/
#   __init__.py
#   model/
#     __init__.py
#     get_model.py
#     get_unlearn_model.py
#   dataset/
#     __init__.py
#     get_cifar10.py
#     get_dataloader.py
#     get_unlearn_dataset.py
#     get_unlearn_dataloader.py
#   utils/
#     __init__.py
#     train.py
#     test.py
#     distiller.py
#     unlearn_distill.py
#   metric/
#     __init__.py
#     mia_attack.py
#     retrain_dataset_acc.py
#     forget_acc.py
#   scripts/
#     train_teacher.py
#     distill_student.py
#     unlearn_kd.py
#     evaluate.py
#   requirements.txt
#   README.md

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

# empty file (package marker)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/model/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

from .get_model import get_model
from .get_unlearn_model import get_unlearn_model

__all__ = ["get_model", "get_unlearn_model"]

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/model/get_model.py
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torchvision.models as models


_CIFAR10_NC = 10


def _patch_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    # CIFAR-10 (32x32): use 3x3 conv, stride=1, no maxpool
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()
    # replace FC head
    if hasattr(model, "fc"):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    return model


def get_model(name: str, num_classes: int = _CIFAR10_NC, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name in ["resnet18", "r18"]:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name in ["resnet34", "r34"]:
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name in ["resnet50", "r50"]:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return _patch_for_cifar(m, num_classes)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/model/get_unlearn_model.py
# ─────────────────────────────────────────────────────────────────────────────

from .get_model import get_model


def get_unlearn_model(name: str, num_classes: int = 10):
    """
    Alias for clarity: returns a student model to be used for unlearning/KD.
    Separated so you can later diverge (e.g., add heads, masks, etc.).
    """
    return get_model(name, num_classes=num_classes, pretrained=False)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/dataset/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

from .get_cifar10 import get_cifar10
from .get_dataloader import get_dataloader
from .get_unlearn_dataset import make_unlearn_retain_split
from .get_unlearn_dataloader import get_unlearn_dataloaders

__all__ = [
    "get_cifar10",
    "get_dataloader",
    "make_unlearn_retain_split",
    "get_unlearn_dataloaders",
]

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/dataset/get_cifar10.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Tuple
import torchvision.transforms as T
from torchvision import datasets


def get_cifar10(data_root: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    # standard augmentation for train; eval transforms for test
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    return train, test

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/dataset/get_dataloader.py
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size=128, num_workers=4, shuffle=True, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/dataset/get_unlearn_dataset.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Subset


class WithFlagSubset(Subset):
    """Subset that exposes a per-sample flag (retain=0 / unlearn=1)."""
    def __init__(self, dataset, indices, flag_value: int):
        super().__init__(dataset, indices)
        self.flag_value = flag_value

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, self.flag_value


def make_unlearn_retain_split(train_dataset, per_class_unlearn_ratio: float = 0.1, seed: int = 42) -> Tuple[WithFlagSubset, WithFlagSubset, Dict[int, Tuple[List[int], List[int]]]]:
    """
    Split CIFAR-10 train set per class so that exactly ratio of samples per class are assigned to UNLEARN set.
    Returns (retain_subset, unlearn_subset, index_map_by_class).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(train_dataset.targets)
    retain_indices: List[int] = []
    unlearn_indices: List[int] = []
    idx_map: Dict[int, Tuple[List[int], List[int]]] = {}

    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        n_unl = max(1, int(round(len(cls_idx) * per_class_unlearn_ratio)))
        u_idx = cls_idx[:n_unl].tolist()
        r_idx = cls_idx[n_unl:].tolist()
        unlearn_indices.extend(u_idx)
        retain_indices.extend(r_idx)
        idx_map[c] = (r_idx, u_idx)

    retain_subset = WithFlagSubset(train_dataset, retain_indices, flag_value=0)
    unlearn_subset = WithFlagSubset(train_dataset, unlearn_indices, flag_value=1)
    return retain_subset, unlearn_subset, idx_map

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/dataset/get_unlearn_dataloader.py
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import ConcatDataset
from .get_dataloader import get_dataloader


def get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=128, num_workers=4):
    retain_loader = get_dataloader(retain_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    unlearn_loader = get_dataloader(unlearn_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    joint_loader = get_dataloader(ConcatDataset([retain_subset, unlearn_subset]), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return retain_loader, unlearn_loader, joint_loader

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/utils/test.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def test(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    model.to(device)
    correct, total = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/utils/train.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from .test import test


def train_teacher(model, train_loader, test_loader, device, epochs=200, lr=0.1, wd=5e-4, momentum=0.9, save_path: str = None) -> Dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    best_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        _, acc = test(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
        scheduler.step()

    # load best back (for immediate use)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_acc": best_acc}

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/utils/distiller.py
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KDConfig:
    T: float = 4.0
    alpha: float = 0.5  # weight on KD; (1-alpha) on CE


def kd_step(student, teacher, batch, device, cfg: KDConfig):
    x, y = batch[:2]  # may include flag
    x, y = x.to(device), y.to(device)
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(x)
    s_logits = student(x)

    # CE (hard)
    ce = F.cross_entropy(s_logits, y)

    # KL (soft)
    T = cfg.T
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = F.softmax(t_logits / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

    loss = cfg.alpha * kl + (1.0 - cfg.alpha) * ce
    return loss


def distill(student, teacher, train_loader, test_loader, device, epochs=100, cfg: KDConfig = KDConfig(), optimizer=None, scheduler=None):
    student.to(device)
    teacher.to(device)

    if optimizer is None:
        optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_acc = 0.0
    best_state = None

    from .test import test

    for ep in range(1, epochs + 1):
        student.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = kd_step(student, teacher, batch, device, cfg)
            loss.backward()
            optimizer.step()
        _, acc = test(student, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
        scheduler.step()

    if best_state is not None:
        student.load_state_dict(best_state)
    return {"best_acc": best_acc}

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/utils/unlearn_distill.py
# ─────────────────────────────────────────────────────────────────────────────

"""
Unlearning KD:
- Retain subset: distill from good_teacher
- Unlearn subset: distill from bad_teacher (e.g., randomly initialized or degraded)
- Joint loader yields (x, y, flag) where flag=0 retain, flag=1 unlearn.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UKDConfig:
    T: float = 4.0
    alpha_retain: float = 0.7  # emphasize retention on retain set
    alpha_unlearn: float = 1.0  # emphasize teacher matching on unlearn set (bad teacher)
    ce_scale_unlearn: float = -0.2  # negative CE to push away true label on forget set (gradient-ascent like)
    lambda_balance: float = 1.0  # scale to balance retain/unlearn if batches mixed


def _kd_loss(s_logits, t_logits, y, T, alpha):
    ce = F.cross_entropy(s_logits, y)
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = F.softmax(t_logits / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
    return alpha * kl + (1 - alpha) * ce


def unlearn_kd_step(student, good_t, bad_t, batch, device, cfg: UKDConfig):
    x, y, flag = batch
    x, y, flag = x.to(device), y.to(device), flag.to(device)

    # split retain/unlearn within the batch
    retain_mask = (flag == 0)
    unlearn_mask = (flag == 1)

    loss = 0.0
    n_parts = 0

    if retain_mask.any():
        xr, yr = x[retain_mask], y[retain_mask]
        with torch.no_grad():
            gt = good_t(xr)
        sr = student(xr)
        loss_r = _kd_loss(sr, gt, yr, cfg.T, cfg.alpha_retain)
        loss = loss + loss_r
        n_parts += 1

    if unlearn_mask.any():
        xu, yu = x[unlearn_mask], y[unlearn_mask]
        with torch.no_grad():
            bt = bad_t(xu)
        su = student(xu)
        # match bad teacher + (optionally) anti-CE to move away from ground-truth
        T = cfg.T
        log_p = F.log_softmax(su / T, dim=1)
        q = F.softmax(bt / T, dim=1)
        kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
        anti_ce = -F.cross_entropy(su, yu)  # gradient ascent on true label
        loss_u = cfg.alpha_unlearn * kl + cfg.ce_scale_unlearn * anti_ce
        loss = loss + cfg.lambda_balance * loss_u
        n_parts += 1

    return loss / max(n_parts, 1)


def unlearn_distill(student, good_teacher, bad_teacher, joint_loader, test_loader, device, epochs=80, cfg: UKDConfig = UKDConfig(), optimizer=None, scheduler=None):
    student.to(device)
    good_teacher.to(device).eval()
    bad_teacher.to(device).eval()

    if optimizer is None:
        optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

    best_acc = 0.0
    best_state = None

    from .test import test

    for ep in range(1, epochs + 1):
        student.train()
        for batch in joint_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = unlearn_kd_step(student, good_teacher, bad_teacher, batch, device, cfg)
            loss.backward()
            optimizer.step()
        _, acc = test(student, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
        scheduler.step()

    if best_state is not None:
        student.load_state_dict(best_state)
    return {"best_acc": best_acc}

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/metric/__init__.py
# ─────────────────────────────────────────────────────────────────────────────

from .mia_attack import mia_success_rate
from .retrain_dataset_acc import eval_retain_acc
from .forget_acc import eval_forget_acc

__all__ = ["mia_success_rate", "eval_retain_acc", "eval_forget_acc"]

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/metric/mia_attack.py
# ─────────────────────────────────────────────────────────────────────────────

"""
Very lightweight MIA proxy: threshold on max softmax confidence.
- Train-time (retain) members tend to have higher confidence than held-out.
- We report attack success on the unlearn subset by pretending they were members
  and seeing how often the attack says "member" (lower is better after unlearning).
Note: for rigorous MIA, plug in a stronger shadow-model attack later.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def mia_success_rate(model, loader_with_flags, device, thresh: float = 0.9) -> Tuple[float, float]:
    model.eval().to(device)
    tp = 0  # predicted member & actually unlearn (treated as positive)
    fp = 0
    fn = 0
    tn = 0

    for x, y, flag in loader_with_flags:
        x = x.to(device)
        logits = model(x)
        conf = F.softmax(logits, dim=1).max(dim=1).values
        pred_member = (conf >= thresh).int().cpu()
        actual_member = flag.int()  # 1 for unlearn samples
        tp += int(((pred_member == 1) & (actual_member == 1)).sum())
        fp += int(((pred_member == 1) & (actual_member == 0)).sum())
        fn += int(((pred_member == 0) & (actual_member == 1)).sum())
        tn += int(((pred_member == 0) & (actual_member == 0)).sum())

    # attack success = TP / (TP+FN); false positive rate = FP / (FP+TN)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return tpr, fpr

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/metric/retrain_dataset_acc.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Tuple
import torch


@torch.no_grad()
def eval_retain_acc(model, retain_loader, device) -> Tuple[float, float]:
    model.eval().to(device)
    correct, total = 0, 0
    import torch.nn.functional as F
    loss_sum = 0.0
    for x, y, _flag in retain_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/metric/forget_acc.py
# ─────────────────────────────────────────────────────────────────────────────

from typing import Tuple
import torch


@torch.no_grad()
def eval_forget_acc(model, unlearn_loader, device) -> Tuple[float, float]:
    """
    Accuracy on UNLEARN set (lower is better if true forgetting is desired).
    Returns (loss, acc).
    """
    model.eval().to(device)
    correct, total = 0, 0
    import torch.nn.functional as F
    loss_sum = 0.0
    for x, y, _flag in unlearn_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/scripts/train_teacher.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse, torch
from unlearn_kd.model import get_model
from unlearn_kd.dataset import get_cifar10, get_dataloader
from unlearn_kd.utils.train import train_teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model(args.model)
    train, test = get_cifar10()
    tr_loader = get_dataloader(train, batch_size=args.bs, shuffle=True)
    te_loader = get_dataloader(test, batch_size=256, shuffle=False)

    stats = train_teacher(m, tr_loader, te_loader, device, epochs=args.epochs, save_path=args.save)
    print({"best_acc": stats["best_acc"]})


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/scripts/distill_student.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse, torch
from unlearn_kd.model import get_model
from unlearn_kd.dataset import get_cifar10, get_dataloader
from unlearn_kd.utils.distiller import distill, KDConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--T", type=float, default=4.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = get_model(args.teacher)
    s = get_model(args.student)
    t.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu"))

    train, test = get_cifar10()
    tr = get_dataloader(train, batch_size=args.bs)
    te = get_dataloader(test, batch_size=256, shuffle=False)

    stats = distill(s, t, tr, te, device, epochs=args.epochs, cfg=KDConfig(T=args.T, alpha=args.alpha))
    print({"best_acc": stats["best_acc"]})


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/scripts/unlearn_kd.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse, torch
from unlearn_kd.model import get_model, get_unlearn_model
from unlearn_kd.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd.utils.unlearn_distill import unlearn_distill, UKDConfig
from unlearn_kd.metric import mia_success_rate, eval_retain_acc, eval_forget_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--good_teacher", type=str, required=True)  # architecture name
    p.add_argument("--bad_teacher", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--good_ckpt", type=str, required=True)
    p.add_argument("--bad_ckpt", type=str, default=None)  # if None -> random init
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.1)  # per-class unlearn ratio
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha_retain", type=float, default=0.7)
    p.add_argument("--alpha_unlearn", type=float, default=1.0)
    p.add_argument("--ce_scale_unlearn", type=float, default=-0.2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # teachers
    good_t = get_model(args.good_teacher)
    good_t.load_state_dict(torch.load(args.good_ckpt, map_location="cpu"))

    if args.bad_ckpt:
        bad_t = get_model(args.bad_teacher)
        bad_t.load_state_dict(torch.load(args.bad_ckpt, map_location="cpu"))
    else:
        bad_t = get_model(args.bad_teacher)  # random init = incompetent teacher

    # student
    student = get_unlearn_model(args.student)

    # data
    train, test = get_cifar10()
    retain_subset, unlearn_subset, _ = make_unlearn_retain_split(train, per_class_unlearn_ratio=args.ratio, seed=42)
    retain_loader, unlearn_loader, joint_loader = get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=args.bs)
    test_loader = get_dataloader(test, batch_size=256, shuffle=False)

    stats = unlearn_distill(
        student, good_t, bad_t, joint_loader, test_loader, device, epochs=args.epochs,
        cfg=UKDConfig(T=args.T, alpha_retain=args.alpha_retain, alpha_unlearn=args.alpha_unlearn, ce_scale_unlearn=args.ce_scale_unlearn)
    )

    # Metrics
    mia_tpr, mia_fpr = mia_success_rate(student, unlearn_loader, device)
    retain_loss, retain_acc = eval_retain_acc(student, retain_loader, device)
    forget_loss, forget_acc = eval_forget_acc(student, unlearn_loader, device)

    print({
        "student_test_best_acc": stats["best_acc"],
        "retain_acc": retain_acc,
        "forget_acc": forget_acc,
        "mia_tpr_on_unlearn": mia_tpr,
        "mia_fpr_on_retain": mia_fpr,
    })


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/scripts/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse, torch
from unlearn_kd.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd.model import get_model
from unlearn_kd.metric import mia_success_rate, eval_retain_acc, eval_forget_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model(args.model)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    train, test = get_cifar10()
    retain_subset, unlearn_subset, _ = make_unlearn_retain_split(train, per_class_unlearn_ratio=args.ratio, seed=42)
    retain_loader, unlearn_loader, _ = get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=args.bs)

    tpr, fpr = mia_success_rate(m, unlearn_loader, device)
    r_loss, r_acc = eval_retain_acc(m, retain_loader, device)
    f_loss, f_acc = eval_forget_acc(m, unlearn_loader, device)

    print({
        "retain_acc": r_acc,
        "forget_acc": f_acc,
        "mia_tpr_on_unlearn": tpr,
        "mia_fpr_on_retain": fpr,
    })


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/requirements.txt
# ─────────────────────────────────────────────────────────────────────────────

torch>=2.2
torchvision>=0.17
numpy

# ─────────────────────────────────────────────────────────────────────────────
# File: unlearn_kd/README.md
# ─────────────────────────────────────────────────────────────────────────────

# unlearn_kd (CIFAR‑10) — Sample Unlearning + Knowledge Distillation

## What this repo gives you
- Teacher training (save best‑acc ckpt)
- Baseline KD (teacher→student) on retain or full set
- Unlearning KD: retain samples distilled from a **good** teacher; unlearn samples distilled from a **bad** teacher (random or weakened)
- Metrics: retain_acc, forget_acc, simple MIA proxy (confidence threshold)

## Quickstart
```bash
# 1) Train teachers
python -m unlearn_kd.scripts.train_teacher --model resnet50 --save ckpt_r50.pt
python -m unlearn_kd.scripts.train_teacher --model resnet34 --save ckpt_r34.pt

# 2) Baseline KD (control): retain‑only
python -m unlearn_kd.scripts.distill_student \
  --teacher resnet50 --student resnet18 --teacher_ckpt ckpt_r50.pt

# 3) Unlearning KD (experiment): retain from good, unlearn from bad
python -m unlearn_kd.scripts.unlearn_kd \
  --good_teacher resnet50 --bad_teacher resnet50 --student resnet18 \
  --good_ckpt ckpt_r50.pt  # --bad_ckpt omit -> random init as bad teacher
```

## Controls vs Experiments
- Control‑1: train teacher on **retain only**, then KD→student on **retain only**.
- Control‑2: train teacher on **full dataset**, then KD→student on **retain only**.
- Experiment: Unlearning KD (retain from good, unlearn from bad) with per‑class 10% forget split (configurable).

## Notes
- All CIFAR‑10 splits are per‑class to keep class balance.
- The unlearning step uses KL to match a bad teacher on forget samples + *anti‑CE* (negative CE) to move away from the ground‑truth.
- Replace `mia_attack.py` with a stronger attack if needed (e.g., shadow models).
