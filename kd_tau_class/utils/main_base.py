# main.py  (수정판: 핵심만 고침)
import torch, torch.nn.functional as F, torch.nn as nn
from model import custom_res18
from dataset import full_forget_retain_loader_train, full_forget_retain_loader_test
from train_test_acc import train_model, model_test
from mia_unlearning_score import UnLearningScore, get_membership_attack_prob, actv_dist
from seed import set_seed

set_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 512
EPOCHS = 30
epoch_unlearn = [k for k in range(1,11)]
LR = 1e-4
TAU_G = [0.1, 0.3, 0.9, 1, 3, 5, 7, 10 ]
TAU_B = [0.1, 0.3, 0.9, 1, 3, 5, 7, 10 ]
FORGET_CLASS = 1
NUM_CLASSES = 10

def kd_loss(student_logits, teacher_logits, T, sign):
    if sign==1:
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
    elif sign==0:
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T )
    else:
        assert sign != 1 and sign !=0,  'error'
        return 0

def main():
    # 1) 데이터
    full_tr, forget_tr, retain_tr = full_forget_retain_loader_train(FORGET_CLASS, batch_size=BATCH)
    full_te, forget_te, retain_te = full_forget_retain_loader_test(FORGET_CLASS, batch_size=BATCH)

    # 2) good teacher 학습 -> 맞고
    good = custom_res18(NUM_CLASSES)
    # good = train_model(good, full_tr.dataset, 'good_teacher', epochs=EPOCHS, lr=LR, batch_size=BATCH, device=DEVICE)
    good.load_state_dict(torch.load('./full_tr_model.pth'))
    good.to(DEVICE)
    good.eval()

    
    retrain = custom_res18(NUM_CLASSES).to(DEVICE)
    retrain.load_state_dict(torch.load('./retrain_tr_model.pth'))
    retrain.eval()

    # 3) bad teacher (랜덤)
    bad = custom_res18(NUM_CLASSES).to(DEVICE).eval()

    # 4) student 초기화 = good 복사
    student = custom_res18(NUM_CLASSES).to(DEVICE)
    student.load_state_dict(good.state_dict())  # ← 중요!
    student.train()
    
    
    

    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    # 5) Dual-τ KD 언러닝
        for TAU_G in tg:
            for TAU_B in tb:
                for epoch_unlearn in epoch_unlearn:
                print(======================================================================)
                print()
                print()
                print(f'unlearng epoch : {epoch_unlearn} tau_g : {TAU_G} tau_b : {TAU_B}')
                for ep in range(epoch_unlearn):
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
                        if mask_r.any():
                            loss += kd_loss(zs[mask_r], zg[mask_r], TAU_G, 1)
                        if mask_f.any():
                            loss += kd_loss(zs[mask_f], zb[mask_f], TAU_B, 0)

                        loss.backward(); opt.step()
                        running += loss.item()
                    print(f'[Unlearn] Epoch {ep+1}/{epoch_unlearn} Loss {running/len(full_tr):.4f}')

                student.eval()

                # 6) 평가 (비교군 포함)
                # 6-0) baseline: student_no_unlearn (= good 사본, 학습 X)
                stu_no_unlearn = custom_res18(NUM_CLASSES).to(DEVICE)
                stu_no_unlearn.load_state_dict(good.state_dict()); stu_no_unlearn.eval()

                #stu_no_unlearn이 good teacher

                forget_eval = torch.utils.data.DataLoader(forget_te.dataset, batch_size=128, shuffle=False)

                print('\n=== Accuracy (추가: Retrain baseline) === retrain model ')
                print('Retrain/Retain_train')
                model_test(retrain, retain_tr.dataset, 'Retrain/Retain_train', batch_size=256, device=DEVICE)
                print('Retrain/Forget_train')
                model_test(retrain, forget_tr.dataset, 'Retrain/Forget_train', batch_size=256, device=DEVICE)
                print('Retrain/Retain-test')
                model_test(retrain, retain_te.dataset, 'Retrain/Retain-test', batch_size=256, device=DEVICE)
                print('Retrain/Forget-test')
                model_test(retrain, forget_te.dataset, 'Retrain/Forget-test', batch_size=256, device=DEVICE)


                print('\n=== Accuracy ===')
                print('Good/Full at full_testset acc' )
                model_test(good,     full_te.dataset,   'Good/Full',   batch_size=256, device=DEVICE)

                print('Student/Retain at retrain_testset acc' )
                model_test(student,  retain_te.dataset, 'Student/Retain', batch_size=256, device=DEVICE)

                print('Student/forget at forget_testset acc' )
                model_test(student,  forget_te.dataset, 'Student/Forget', batch_size=256, device=DEVICE)

                print('Student/Retain at retrain_trainset acc' )
                model_test(student,  retain_tr.dataset, 'Student/Retain', batch_size=256, device=DEVICE)

                print('Student/forget at forget_trainset acc' )
                model_test(student,  forget_tr.dataset, 'Student/Forget', batch_size=256, device=DEVICE)


                # 6-1) UL Score (vs Bad Teacher)  (★ 대상 교정)
                print('forget_test를 이용한 ulscore 계산 zrf 분포 이용 논문 인용 분포가 얼마나 비슷한가')
                forget_eval = torch.utils.data.DataLoader(forget_tr.dataset, batch_size=128, shuffle=False)
                zrf = UnLearningScore(student, bad, forget_eval, DEVICE)
                print(f'ZRF (1 - JS w/ bad) : {zrf:.4f} studuent badteacher 비교')


                zrf_retrain = UnLearningScore(retrain, bad, forget_eval, DEVICE)
                print(f'ZRF Retrain (1 - JS vs bad): {zrf_retrain:.4f} retrain badteacher 비교')

                # 6-2) MIA (낮을수록 좋음)
                retain_train_eval = torch.utils.data.DataLoader(retain_tr.dataset, batch_size=128, shuffle=False)
                retain_test_eval  = torch.utils.data.DataLoader(retain_te.dataset, batch_size=128, shuffle=False)
                mia_p = get_membership_attack_prob(retain_train_eval, forget_eval, retain_test_eval, student, DEVICE)
                print(f'MIA success prob on Forget: {mia_p:.4f}')

                # 6-3) Activation distance (good vs student on forget)
                act = actv_dist(good, student, forget_eval, DEVICE)
                print(f'Activation distance (good vs student, forget_eval): {act:.4f}')

                act1 = actv_dist(retrain, student, forget_eval, DEVICE)
                print(f'Activation distance (retrain vs student, forget_eval): {act1:.4f}')

if __name__ == '__main__':
    main()
