import argparse, torch
from unlearn_kd_sample.model import get_model, get_unlearn_model
from unlearn_kd_sample.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd_sample.utils.unlearn_distill import unlearn_distill, UKDConfig
from unlearn_kd_sample.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc
import pandas as pd
from unlearn_kd_sample.utils.seed import set_seed


def main():
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--good_teacher", type=str, required=True)  # architecture name
    p.add_argument("--bad_teacher", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--good_ckpt", type=str, required=True)
    p.add_argument("--bad_ckpt", type=str, default=None)  # if None -> random init
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.05)  # per-class unlearn ratio
    p.add_argument("--lambda_balance", type=float, default=0.5)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha_retain", type=float, default=0.7)
    p.add_argument("--alpha_unlearn", type=float, default=1.0)
    p.add_argument("--ce_scale_unlearn", type=float, default=0.0)
    p.add_argument("--student_pretrained", type=str, default=False)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--device", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # teachers
    good_t = get_model(args.good_teacher)
    good_t.load_state_dict(torch.load(args.good_ckpt, map_location="cpu"))

    if args.bad_ckpt:
        bad_t = get_model(args.bad_teacher)
        bad_t.load_state_dict(torch.load(args.bad_ckpt, map_location="cpu"))
    else:
        bad_t = get_model(args.bad_teacher)  # random init = incompetent teacher

    # student
    if args.student_pretrained == 'True':
        print("Using pretrained weights for student model")
        student = get_model(args.student)
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))  # load good teacher weights
    else:
        print("Using random init for student model")
        student = get_unlearn_model(args.student, 10, False)

    # data
    train, test = get_cifar10()
    test_loader = get_dataloader(test, batch_size=256, shuffle=False)
    retain_subset, unlearn_subset, _ = make_unlearn_retain_split(train, per_class_unlearn_ratio=args.ratio, seed=42)
    retain_loader, unlearn_loader, joint_loader = get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=args.bs)

    

    # unlearn distill
    result_of_unlearn_distill = unlearn_distill(
        student, good_t, bad_t, joint_loader, test_loader, unlearn_loader, retain_loader, device, epochs=args.epochs,
        cfg=UKDConfig(teacher = args.good_teacher, student = args.student, T=args.T, alpha_retain=args.alpha_retain, alpha_unlearn=args.alpha_unlearn, 
        ce_scale_unlearn=args.ce_scale_unlearn, lambda_balance=args.lambda_balance)
    )
    stats, metrics = result_of_unlearn_distill["best_acc"], result_of_unlearn_distill["dic_metrics"]
    # Metrics
    mia_rate = get_membership_attack_prob(retain_loader, unlearn_loader, test_loader, student, device)
    retain_loss, retain_acc = eval_retain_acc(student, retain_loader, device)
    forget_loss, forget_acc = eval_forget_acc(student, unlearn_loader, device)

    print({
        "student_test_best_acc": stats["best_acc"],
        "retain_acc": retain_acc,
        "forget_acc": forget_acc,
        "mia_rate": mia_rate,

    })

if __name__ == "__main__":
    main()