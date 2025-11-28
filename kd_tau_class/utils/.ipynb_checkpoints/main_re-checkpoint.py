# main.py  (정리/수정판)
import torch, torch.nn.functional as F, torch.nn as nn
from model import custom_res18
from dataset import full_forget_retain_loader_train, full_forget_retain_loader_test
from train_test_acc import train_model, model_test
from mia_unlearning_score import UnLearningScore, get_membership_attack_prob, actv_dist
from seed import set_seed
import gc

# ===== 고정 하이퍼 =====
set_seed(42)
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
BATCH = 512
# EPOCHS = 30              # good teacher 학습시 쓰는 값 (현재는 load_state_dict)
LR = 1e-4
FORGET_CLASS = 1
NUM_CLASSES = 10

# ===== 실험 하이퍼 (그리드) =====
TAU_G_LIST = [0.01, 0.25, 0.5, 0.75,  2,  4, 100]   # retain 쪽에서 good teacher와 KD할 온도
TAU_B_LIST = [0.01, 0.25, 0.5, 0.75,  2,  4, 100]   # forget 쪽에서 bad teacher와 KD할 온도
EPOCH_COUNTS = [1, 5]              # 언러닝 반복 에폭
CE_TOGGLE = [False, True]                      # CrossEntropy on/off (retain 샘플에만 적용)

# ===== KD 손실 =====
def kd_loss(student_logits, teacher_logits, T, scale_T2=True):
    """
    기본 KD: KL( softmax(z_s/T), softmax(z_t/T) ) * (T^2)  (Hinton 권장 스케일)
    scale_T2=False로 주면 T만 곱하거나 스케일 없이도 실험 가능하게 확장하려면 아래를 바꿔도 됨.
    """
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    )
    return max(T,(T ** 2)) * kl if scale_T2 else kl

def main():
    # 1) 데이터
    full_tr, forget_tr, retain_tr = full_forget_retain_loader_train(FORGET_CLASS, batch_size=BATCH)
    full_te, forget_te, retain_te = full_forget_retain_loader_test(FORGET_CLASS, batch_size=BATCH)

    # 2) good / retrain / bad teacher 준비
    good = custom_res18(NUM_CLASSES)
    # 필요시 학습: good = train_model(...)
    good.load_state_dict(torch.load('./full_tr_model.pth', map_location='cpu'))
    good.to(DEVICE).eval()

    retrain = custom_res18(NUM_CLASSES).to(DEVICE)
    retrain.load_state_dict(torch.load('./retrain_tr_model.pth', map_location='cpu'))
    retrain.eval()

    # 랜덤 bad teacher (고정)
    bad = custom_res18(NUM_CLASSES).to(DEVICE).eval()

    # ===== Retrain 베이스라인 (1회만 출력) =====
    print('\n=== Accuracy (Retrain baseline, 1회) ===')
    model_test(retrain, retain_tr.dataset, 'Retrain/Retain_train', batch_size=256, device=DEVICE)
    model_test(retrain, forget_tr.dataset, 'Retrain/Forget_train', batch_size=256, device=DEVICE)
    model_test(retrain, retain_te.dataset, 'Retrain/Retain_test',  batch_size=256, device=DEVICE)
    model_test(retrain, forget_te.dataset, 'Retrain/Forget_test',  batch_size=256, device=DEVICE)

    print('\n=== Good Teacher baseline (1회) ===')
    model_test(good, full_te.dataset, 'Good/Full_test', batch_size=256, device=DEVICE)

    # 평가용 로더(혼동 없게 train/test 분리)
    forget_eval_train = torch.utils.data.DataLoader(forget_tr.dataset,  batch_size=128, shuffle=False)
    forget_eval_test  = torch.utils.data.DataLoader(forget_te.dataset,  batch_size=128, shuffle=False)
    retain_eval_train = torch.utils.data.DataLoader(retain_tr.dataset,  batch_size=128, shuffle=False)
    retain_eval_test  = torch.utils.data.DataLoader(retain_te.dataset,  batch_size=128, shuffle=False)

    print('\n' + '='*88)
    print(' Grid Search: (tau_g × tau_b × epochs × CE_on) — 언러닝 성능 점검 ')
    print('='*88 + '\n')

    # 3) 그리드 서치
    # for tg in TAU_G_LIST:
    for ce_on in CE_TOGGLE:
        for tg in TAU_G_LIST:
        # for ce_on in CE_TOGGLE:
            for tb in TAU_B_LIST:
                for E in EPOCH_COUNTS:
                    

                    # (중요) 실험마다 student를 good으로 초기화 + 옵티마이저 리셋 + seed 고정
                    set_seed(42)
                    student = custom_res18(NUM_CLASSES).to(DEVICE)
                    student.load_state_dict(good.state_dict())
                    student.train()

                    opt = torch.optim.AdamW(student.parameters(), lr=LR)

                    print('='*70)
                    print(f'[Unlearn Run] epochs={E:2d} | tau_g={tg:>4} | tau_b={tb:>4} | CE_on={ce_on}')
                    for ep in range(E):
                        running = 0.0
                        for x, y in full_tr:
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            opt.zero_grad()
                            zs = student(x)
                            with torch.no_grad():
                                zg = good(x)
                                zb = bad(x)

                            mask_f = (y == FORGET_CLASS)
                            mask_r = ~mask_f

                            loss = torch.tensor(0., device=DEVICE)

                            # retain → good teacher KD
                            if mask_r.any():
                                loss = loss + kd_loss(zs[mask_r], zg[mask_r], T=tg, scale_T2=True)
                                # CE on/off (retain 샘플에만 적용)
                                if ce_on:
                                    loss = loss + F.cross_entropy(zs[mask_r], y[mask_r])

                            # forget → bad teacher KD
                            if mask_f.any():
                                # 표준대로면 여기서도 T^2 스케일을 붙이는 게 일관적이지만,
                                # 사용자가 기존 코드에서 다른 스케일을 시도했으므로 그대로 T^2 사용 (필요시 바꿔보세요).
                                loss = loss + kd_loss(zs[mask_f], zb[mask_f], T=tb, scale_T2=True)
                                if ce_on:
                                    loss = loss + F.cross_entropy(zs[mask_r], y[mask_r])


                            loss.backward()
                            opt.step()
                            running += loss.item()

                        print(f'  Epoch {ep+1:2d}/{E:2d}  Loss {running/len(full_tr):.4f}')

                    # ===== 평가 =====
                    student.eval()

                    print('\n[Accuracies]')
                    model_test(student, retain_te.dataset, 'Student/Retain_test', batch_size=256, device=DEVICE)
                    model_test(student, forget_te.dataset, 'Student/Forget_test', batch_size=256, device=DEVICE)
                    model_test(student, retain_tr.dataset, 'Student/Retain_train', batch_size=256, device=DEVICE)
                    model_test(student, forget_tr.dataset, 'Student/Forget_train', batch_size=256, device=DEVICE)

                    # UL Score (forget_test 사용으로 통일)
                    print('\n[UL Score / JS 기반]')
                    zrf = UnLearningScore(student, bad, forget_eval_test, DEVICE)
                    print(f'  ZRF (1 - JS vs bad) on forget_test : {zrf:.4f}')

                    # Retrain과도 비교 (참고용)
                    zrf_retrain = UnLearningScore(retrain, bad, forget_eval_test, DEVICE)
                    print(f'  ZRF Retrain (1 - JS vs bad)       : {zrf_retrain:.4f}')

                    # MIA (낮을수록 좋음) — retain_train / forget_test / retain_test
                    print('\n[MIA]')
                    mia_p = get_membership_attack_prob(retain_eval_train, forget_eval_test, retain_eval_test, student, DEVICE)
                    print(f'  MIA success prob on Forget (test) : {mia_p:.4f}')

                    # Activation distance
                    print('\n[Activation distance]')
                    act_gs = actv_dist(good,    student, forget_eval_test, DEVICE)
                    act_rs = actv_dist(retrain, student, forget_eval_test, DEVICE)
                    print(f'  good vs student (forget_test)     : {act_gs:.4f}')
                    print(f'  retrain vs student (forget_test)  : {act_rs:.4f}\n')

                    # 메모리 정리 (긴 그리드 돌릴 때 OOM 방지)
                    del student, opt
                    torch.cuda.empty_cache()
                    gc.collect()

    print('=== All experiments finished. ===')

if __name__ == '__main__':
    main()
