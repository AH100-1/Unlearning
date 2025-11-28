from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from unlearn_kd_sample.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc


@dataclass
class UKDConfig:
    teacher : str = "resnet50"  # architecture name for both good and bad teacher
    student : str = "resnet18"
    T: float = 4.0
    alpha_retain: float = 0.7  # at retain alpha for KD vs CE
    alpha_unlearn: float = 1.0  # constant of kd at unlearn
    ce_scale_unlearn: float = 0.2  # negative CE to push away true label on forget set (gradient-ascent like)
    lambda_balance: float = 0.5  # re conconstant of kd before calculating total loss

    
def _kd_loss(s_logits, t_logits, y, T, alpha):
    ce = F.cross_entropy(s_logits, y)
    log_p = F.log_softmax(s_logits / T, dim=1)
    q = F.softmax(t_logits / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)
    return alpha * kl + (1 - alpha) * ce


def unlearn_kd_step(student, good_t, bad_t, batch, device, ep, cfg: UKDConfig):
    x, y, flag = batch
    x, y, flag = x.to(device), y.to(device), flag.to(device)

    retain_mask = (flag == 0)
    unlearn_mask = (flag == 1)

    loss = torch.tensor(0.0, device=device)
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

      
        lambda_balance = cfg.lambda_balance
       
        ramp = [0.6, 0.8, 1]
        ramped = [r * lambda_balance for r in ramp]

        su = student(xu)
        T = 1.0
        log_p = F.log_softmax(su / T, dim=1)
        q = F.softmax(bt / T, dim=1)
        kl = F.kl_div(log_p, q, reduction="batchmean")
        anti_ce = -F.cross_entropy(su, yu)
        loss_u = cfg.alpha_unlearn * kl + cfg.ce_scale_unlearn * anti_ce

        # 에폭 스케줄 (<=20은 기본값 유지)
        if 20 < ep <= 30:
            lambda_balance = ramped[0]
        elif 30 < ep <= 40:
            lambda_balance = ramped[1]
        elif ep > 40:
            lambda_balance = ramped[2]
        # else: ep <= 20 -> cfg.lambda_balance 그대로

        loss = loss + lambda_balance * loss_u
        n_parts += 1

    return loss / max(n_parts, 1)


def unlearn_distill(student, good_teacher, bad_teacher, joint_loader, test_loader, unlearn_loader, retain_loader, device, epochs=50, cfg: UKDConfig = UKDConfig(), optimizer=None, scheduler=None):
    student.to(device)
    good_teacher.to(device).eval()
    bad_teacher.to(device).eval()

    #cfg attri
    dic_metrics = {
        "good_teacher": cfg.teacher,
        "student": cfg.student,
        'T': cfg.T,
        "alpha_retain": cfg.alpha_retain,
        "alpha_unlearn": cfg.alpha_unlearn,
        "ce_scale_unlearn": cfg.ce_scale_unlearn,
        "lambda_balance": cfg.lambda_balance,
        "forget_acc": [],
        "retain_acc": [],
        "mia_rate": [],
        
    }

    if optimizer is None:
        optimizer = torch.optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.5)

    best_acc = 0.0
    best_state = None

    from .test import test

    for ep in range(1, epochs + 1):
        student.train()

        if ep <= 20:
            for batch in retain_loader:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    t_logits = good_teacher(x)
                s_logits = student(x)

                loss = _kd_loss(s_logits, t_logits, y, cfg.T, cfg.alpha_retain)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        else:
            for batch in retain_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = unlearn_kd_step(student, good_teacher, bad_teacher, batch, device, ep, cfg)
                loss.backward()
                optimizer.step()
            
            for batch in unlearn_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = unlearn_kd_step(student, good_teacher, bad_teacher, batch, device, ep, cfg)
                loss.backward()
                optimizer.step()

        
        _, acc = test(student, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in student.state_dict().items()}
        
        mia_rate = get_membership_attack_prob(retain_loader, unlearn_loader, test_loader, student, device)
        retain_loss, retain_acc = eval_retain_acc(student, retain_loader, device)
        forget_loss, forget_acc = eval_forget_acc(student, unlearn_loader, device)
        print(f'teacher: {cfg.teacher}, student: {cfg.student}')
        print(f"Epoch {ep:03d}: test_acc={acc:.2f}%, best_acc={best_acc:.2f}%, retain_acc={retain_acc:.2f}%, forget_acc={forget_acc:.2f}%, mia_rate={mia_rate:.2f}%")

        scheduler.step()

        dic_metrics["forget_acc"].append(forget_acc)
        dic_metrics["retain_acc"].append(retain_acc)
        dic_metrics["mia_rate"].append(mia_rate)

    path_metric = '/data/khw/unlearn_kd/lab_metric'
    torch.save(dic_metrics,path_metric+'/'+cfg.teacher+'_'+cfg.student+'_metric.pkl')
    if best_state is not None:
        student.load_state_dict(best_state)
    return {"best_acc": best_acc, "dic_metrics": dic_metrics}