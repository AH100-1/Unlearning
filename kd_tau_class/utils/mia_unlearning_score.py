import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def js_divergence(p, q, eps=1e-12, reduction='mean'):
    p = torch.clamp(p, eps, 1.); q = torch.clamp(q, eps, 1.)
    m = 0.5*(p+q)
    kl_pm = F.kl_div(p.log(), m, reduction='batchmean' if reduction=='mean' else 'none')
    kl_qm = F.kl_div(q.log(), m, reduction='batchmean' if reduction=='mean' else 'none')
    return 0.5*(kl_pm + kl_qm)

@torch.no_grad()
def UnLearningScore(student, bad_teacher, forget_loader, device):
    # ZRF-like: student 분포가 bad teacher 분포와 가까울수록 ↑
    student.eval(); bad_teacher.eval()
    preds_s, preds_b = [], []
    for batch in forget_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, _ = batch[:2]
        else:
            x = batch
        x = x.to(device)
        ps = F.softmax(student(x), dim=1)
        pb = F.softmax(bad_teacher(x), dim=1)
        preds_s.append(ps.cpu()); preds_b.append(pb.cpu())
    ps = torch.cat(preds_s); pb = torch.cat(preds_b)
    zrf = 1.0 - js_divergence(ps, pb)
    return float(zrf.item())

@torch.no_grad()
def entropy(p, dim=-1):
    p = torch.clamp(p, 1e-12, 1.0)
    return -(p * p.log()).sum(dim=dim)  # [N]

@torch.no_grad()
def collect_prob(loader, model, device):
    model.eval()
    probs = []
    for batch in loader:
        x, y = batch[:2] if isinstance(batch, (list, tuple)) else (batch, None)
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=-1)       # 평가 시 T=1 고정
        probs.append(p.detach().cpu())       # CPU로 모아 numpy 변환 안전
    return torch.cat(probs, dim=0)           # [N, C]

@torch.no_grad()
def get_membership_attack_prob(retain_train_loader, forget_loader, retain_test_loader, model, device):
    # member vs non-member 학습용 확률 수집
    p_mem    = collect_prob(retain_train_loader, model, device)   # [N_mem, C]
    p_non    = collect_prob(retain_test_loader,  model, device)   # [N_non, C]
    p_forget = collect_prob(forget_loader,       model, device)   # [N_fgt, C]

    # 1D feature: entropy
    H_mem = entropy(p_mem).unsqueeze(1)       # [N_mem, 1]
    H_non = entropy(p_non).unsqueeze(1)       # [N_non, 1]
    H_fgt = entropy(p_forget).unsqueeze(1)    # [N_fgt, 1]

    Xr = torch.cat([H_mem, H_non], dim=0).cpu().numpy()  # [N_mem+N_non, 1]
    Yr = np.concatenate([np.ones(len(H_mem)), np.zeros(len(H_non))])
    Xf = H_fgt.cpu().numpy()

    # 스케일링(권장: RBF SVM은 스케일 민감)
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    Xf_s = scaler.transform(Xf)

    # 확률 평균이 0/1 평균보다 정보를 더 많이 줌
    clf = SVC(C=3, gamma='auto', kernel='rbf', probability=True, random_state=42)
    clf.fit(Xr_s, Yr)

    mia_prob = clf.predict_proba(Xf_s)[:, 1].mean()  # forget의 member 확률 평균 (낮을수록 좋음)
    return float(mia_prob)

@torch.no_grad()
def actv_dist(model1, model2, dataloader, device):
    model1.eval(); model2.eval()
    dists = []
    for batch in dataloader:
        x, y = batch[:2] if isinstance(batch, (list, tuple)) else (batch, None)
        x = x.to(device)
        p1 = F.softmax(model1(x), dim=1)
        p2 = F.softmax(model2(x), dim=1)
        diff = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=1)).cpu()
        dists.append(diff)
    return float(torch.cat(dists).mean().item())
