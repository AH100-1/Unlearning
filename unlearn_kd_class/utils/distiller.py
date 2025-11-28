from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KDConfig:
    T: float = 2.0
    alpha: float = 0.7  # weight on KD; (1-alpha) on CE


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


def distill(student, teacher, train_loader, test_loader, device, epochs=50, cfg: KDConfig = KDConfig(), optimizer=None, scheduler=None, save=None):
    student.to(device)
    teacher.to(device)

    if optimizer is None:
        optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    # if scheduler is None:
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

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
        # scheduler.step()

    if best_state is not None:
        student.load_state_dict(best_state)
        if save is not None:
            torch.save(best_state, save)
    return {"best_acc": best_acc}