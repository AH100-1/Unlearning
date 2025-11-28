import argparse, torch
from unlearn_kd_class.model import get_model, get_unlearn_model
from unlearn_kd_class.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd_class.utils.unlearn_distill import unlearn_distill, UKDConfig
from unlearn_kd_class.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc
import pandas as pd
from unlearn_kd_class.utils.seed import set_seed


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
    p.add_argument("--class_target", type=int, default=0)  # per-class unlearn ratio
    p.add_argument("--g_T", type=float, default=4.0)
    p.add_argument("--b_T", type=float, default=4.0)
    p.add_argument("--alpha_retain", type=float, default=0.7)
    p.add_argument("--alpha_unlearn", type=float, default=0.7)
    p.add_argument("--lambda_balance", type=float, default=1)
    p.add_argument("--ce_scale_unlearn", type=float, default=0.3)
    p.add_argument("--student_pretrained", type=str, default="False")
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
    if args.student_pretrained == "True":
        print("Using pretrained weights for student model")
        student = get_model(args.student)
        student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))  # load good teacher weights
    else:
        print("Using random init for student model")
        student = get_unlearn_model(args.student, 10, False)

    # data
    train, test = get_cifar10()
    train_loader = get_dataloader(train, batch_size=args.bs, shuffle=True)
    test_loader = get_dataloader(test, batch_size=256, shuffle=False)
    retain_subset_train, unlearn_subset_train = make_unlearn_retain_split(train, class_target=0, seed=42)
    retain_subset_test, unlearn_subset_test = make_unlearn_retain_split(test, class_target=0, seed=42)

    retain_loader_train, unlearn_loader_train = get_unlearn_dataloaders(retain_subset_train, unlearn_subset_train, batch_size=args.bs)
    retain_loader_test, unlearn_loader_test = get_unlearn_dataloaders(retain_subset_test, unlearn_subset_test, batch_size=256)

    # unlearn distill
    dic_unlearn_distill = unlearn_distill(
        student, good_t, bad_t, train_loader, test_loader, unlearn_loader_train, retain_loader_train,  retain_loader_test, unlearn_loader_test, device, epochs=args.epochs,
        cfg=UKDConfig(teacher = args.good_teacher, student = args.student, g_T=args.g_T, b_T=args.b_T, alpha_retain=args.alpha_retain, 
        alpha_unlearn=args.alpha_unlearn, lambda_balance=args.lambda_balance, ce_scale_unlearn=args.ce_scale_unlearn, class_target=args.class_target)
    )

    stats = dic_unlearn_distill["stats"]
    dic_metrics = dic_unlearn_distill["metrics"]
    # Metrics
    mia_rate = get_membership_attack_prob(retain_loader_train, unlearn_loader_test, retain_loader_test, student, device)
    retain_loss_train, retain_acc_train = eval_retain_acc(student, retain_loader_train, device)
    forget_loss_train, forget_acc_train = eval_forget_acc(student, unlearn_loader_train, device)
    #metrics of test set
    retain_loss_test, retain_acc_test = eval_retain_acc(student, retain_loader_test, device)
    forget_loss_test, forget_acc_test = eval_forget_acc(student, unlearn_loader_test, device)

    print({
        "student_test_best_acc": stats["best_acc"],
        "retain_acc_train": retain_acc_train,
        "forget_acc_train": forget_acc_train,
        "mia_rate": mia_rate,
        "retain_acc_test": retain_acc_test,
        "forget_acc_test": forget_acc_test,
    })

if __name__ == "__main__":
    main()