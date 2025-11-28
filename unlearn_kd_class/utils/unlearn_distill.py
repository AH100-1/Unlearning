from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from unlearn_kd_class.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc


@dataclass
class UKDConfig:
    teacher: str = "resnet50"  # architecture name for both good and bad teacher
    student: str = "resnet18"
    g_T: float = 4.0
    b_T: float = 4.0
    alpha_retain: float = 0.7  # at retain alpha for KD vs CE
    alpha_unlearn: float = 1.0  # constant of kd at unlearn
    ce_scale_unlearn: float = 0.2  # negative CE to push away true label on forget set (gradient-ascent like)
    lambda_balance: float = 1  # constant of kd before calculating total loss
    class_target: int = 0  # which class to unlearn


def _kd_loss(s_logits, t_logits, y, T, alpha):
    ce = F.cross_entropy(s_logits, y)
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = F.softmax(t_logits / T, dim=1)
    if T >= 1:
        kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
    else:
        kl = F.kl_div(log_p, q, reduction="batchmean") * (T)
    return alpha * kl + (1 - alpha) * ce


def unlearn_kd_step(student, good_t, bad_t, batch, device, cfg: UKDConfig):
    x, y = batch
    x, y = x.to(device), y.to(device)

    # split retain/unlearn within the batch
    retain_mask = (y != 0)
    unlearn_mask = (y == 0)

    loss = 0.0
    n_parts = 0

    if retain_mask.any():
        xr, yr = x[retain_mask], y[retain_mask]
        with torch.no_grad():
            gt = good_t(xr)
        sr = student(xr)
        loss_r = _kd_loss(sr, gt, yr, cfg.g_T, cfg.alpha_retain)
        loss += loss_r
        n_parts += 1

    if unlearn_mask.any():
        xu, yu = x[unlearn_mask], y[unlearn_mask]
        with torch.no_grad():
            bt = bad_t(xu)  # Bad teacher (no gradient required)

        su = student(xu)  # Student forward pass

        # Match bad teacher + (optionally) anti-CE to move away from ground-truth
        T = cfg.b_T  # no temperature scaling for unlearning

        # KL divergence between student and bad teacher (no uniform logits needed)
        log_p = F.log_softmax(su / T, dim=1)
        q = F.softmax(bt / T, dim=1)
        if T >= 1:
            kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
        else:
            kl = F.kl_div(log_p, q, reduction="batchmean") * (T)

        # Anti-CE to push away from true labels (gradient ascent on true label)
        anti_ce = -F.cross_entropy(su, yu)

        # Unlearning loss
        loss_u = cfg.alpha_unlearn * kl + cfg.ce_scale_unlearn * anti_ce

        # Final loss
        loss += cfg.lambda_balance * loss_u
        n_parts += 1

    return loss / max(n_parts, 1)


def unlearn_distill(student, good_teacher, bad_teacher, train_loader, test_loader, unlearn_loader, retain_loader,
                    retain_loader_test, unlearn_loader_test, device, epochs=50,
                    cfg: UKDConfig = UKDConfig(), optimizer=None, scheduler=None):
    student.to(device)
    good_teacher.to(device).eval()
    bad_teacher.to(device).eval()

    dic_metrics = {
        "good_teacher": cfg.teacher,
        "student": cfg.student,
        'gT': cfg.g_T,
        'bT': cfg.b_T,
        "alpha_retain": cfg.alpha_retain,
        "alpha_unlearn": cfg.alpha_unlearn,
        "ce_scale_unlearn": cfg.ce_scale_unlearn,
        "lambda_balance": cfg.lambda_balance,
        "forget_acc": [],
        "retain_acc": [],
        "mia_rate": [],
        "retain_acc_test": [],
        "forget_acc_test": []
    }

    if optimizer is None:
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)


    best_acc = 0.0
    best_state = None

    from .test import test

    for ep in range(1, epochs + 1):
        student.train()

        # Process retain_loader_train
        for batch in retain_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = unlearn_kd_step(student, good_teacher, bad_teacher, batch, device, cfg)
            loss.backward()
            optimizer.step()

        # Process unlearn_loader_train
        for batch in unlearn_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = unlearn_kd_step(student, good_teacher, bad_teacher, batch, device, cfg)
            loss.backward()
            optimizer.step()

        # Test after each epoch
        _, acc = test(student, retain_loader_test, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}

        # Calculate and print metrics
        mia_rate = get_membership_attack_prob(retain_loader, unlearn_loader_test, retain_loader_test, student, device)
        retain_loss, retain_acc = eval_retain_acc(student, retain_loader, device)
        forget_loss, forget_acc = eval_forget_acc(student, unlearn_loader, device)
        retain_loss, retain_acc_test = eval_retain_acc(student, retain_loader_test, device)
        forget_loss, forget_acc_test = eval_forget_acc(student, unlearn_loader_test, device)

        print(f'teacher: {cfg.teacher}, student: {cfg.student}')
        print(f"Epoch {ep:03d}: retain_test_acc={retain_acc_test:.2f}%, retain_test_best_acc={best_acc:.2f}%, forget_acc_test={forget_acc_test:.2f}%, "
              f"retain_acc={retain_acc:.2f}%, forget_acc={forget_acc:.2f}%, mia_rate={mia_rate:.2f}%")

        scheduler.step()

        dic_metrics["forget_acc"].append(forget_acc)
        dic_metrics["retain_acc"].append(retain_acc)
        dic_metrics["retain_acc_test"].append(retain_acc_test)
        dic_metrics["forget_acc_test"].append(forget_acc_test)
        dic_metrics["mia_rate"].append(mia_rate)

    # Save metrics and best model state
    path_metric = '/data/khw/unlearn_kd_class/lab_metric'
    torch.save(dic_metrics, path_metric + '/' + cfg.teacher + '_' + cfg.student + '_metric.pkl')

    if best_state is not None:
        student.load_state_dict(best_state)

    return {"best_acc": best_acc, "dic_metrics": dic_metrics}
