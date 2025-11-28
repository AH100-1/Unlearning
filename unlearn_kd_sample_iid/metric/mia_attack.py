import torch
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Subset, DataLoader

def balance_loader_by_stratified_sampling(retain_loader, test_loader, batch_size=32):
    # retain_loader와 test_loader의 크기
    retain_size = len(retain_loader.dataset)
    test_size = len(test_loader.dataset)

    # retain_loader의 라벨 데이터를 추출
    labels = torch.tensor([label for _, label, flag in retain_loader.dataset])

    # StratifiedShuffleSplit 객체 생성 (retain_loader에서 test_loader 크기만큼 stratified sampling)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    # stratified sampling (retain_loader에서 test_loader 크기만큼 샘플링)
    for _, sampled_idx in sss.split(np.zeros(len(labels)), labels):
        sampled_indices = sampled_idx

    # 샘플링된 인덱스를 retain_loader에 적용
    sampled_subset = Subset(retain_loader.dataset, sampled_indices)

    # 새롭게 샘플링된 subset을 데이터로더로 래핑
    retain_train_loader = DataLoader(sampled_subset, batch_size=1, shuffle=False)

    return retain_train_loader

@torch.no_grad()
def collect_prob(loader, model, device):
    """모델 softmax 확률 벡터 수집"""
    model.eval().to(device)
    probs = []
    for batch in loader:
        # flag 있는 경우
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        else:  # flag 없는 경우
            x, y = batch

        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1).cpu()
        probs.append(p)
    return torch.cat(probs, dim=0)  # [N, C]

def entropy(p: torch.Tensor) -> torch.Tensor:
    # 로그 안정화와 clamp
    log_p = (p + 1e-12).log()
    H = -(p * log_p).sum(dim=1)
    # 만약 H가 상수면 아주 작은 노이즈 추가
    H += 1e-8 * torch.randn_like(H)
    return H
    
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_membership_attack_prob(retain_train_loader, forget_loader,
                               test_loader, model, device) -> float:
    """
    Shadow 없이도 실행 가능한 entropy 기반 MIA
    - retain_train_loader → member (label=1)
    - test_loader  → non-member (label=0)
    - forget_loader       → 평가 대상 (언러닝 샘플)
    반환값: forget set의 'member' 확률 평균 (낮을수록 언러닝 잘 됨)

    """

    retrain_train_loader = balance_loader_by_stratified_sampling(retain_train_loader, test_loader, batch_size=1)
    test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
    forget_loader = DataLoader(forget_loader.dataset, batch_size=1, shuffle=False)

    # 1) 확률 수집
    p_mem    = collect_prob(retain_train_loader, model, device)
    p_non    = collect_prob(test_loader,  model, device)
    p_forget = collect_prob(forget_loader, model, device)

    # 2) feature: entropy
    # 2) feature: entropy
    H_mem = entropy(p_mem).unsqueeze(1)   # [N_mem, 1]
    H_non = entropy(p_non).unsqueeze(1)   # [N_non, 1]
    H_fgt = entropy(p_forget).unsqueeze(1)

    # 3) 학습 데이터 준비 (member=1, non-member=0)
    Xr = torch.cat([H_mem, H_non], dim=0).cpu().numpy()
    Yr = np.concatenate([np.ones(len(H_mem)), np.zeros(len(H_non))])
    Xf = H_fgt.cpu().numpy()

    # 4) 스케일링 + 안정화
    scaler = StandardScaler()
    eps = 1e-8

    # 아주 작은 노이즈 추가 → 분산 확보
    Xr += eps * np.random.randn(*Xr.shape)
    Xf += eps * np.random.randn(*Xf.shape)

    # 스케일링
    Xr_s = scaler.fit_transform(Xr)
    Xf_s = scaler.transform(Xf)

    # nan, inf 제거
    Xr_s = np.nan_to_num(Xr_s, nan=eps, posinf=eps, neginf=-eps)
    Xf_s = np.nan_to_num(Xf_s, nan=eps, posinf=eps, neginf=-eps)




    # 5) 공격자 classifier 학습 (SVM)
    clf = SVC(C=1, gamma="scale", kernel="rbf", probability=True, random_state=42)
    clf.fit(Xr_s, Yr)

    # 6) forget set 평가
    mia_prob = clf.predict_proba(Xf_s)[:, 1].mean()
    return float(mia_prob)
