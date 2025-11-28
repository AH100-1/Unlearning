import argparse, torch
from unlearn_kd_sample.dataset import get_cifar10, make_unlearn_retain_split, get_unlearn_dataloaders, get_dataloader
from unlearn_kd_sample.model import get_model
from unlearn_kd_sample.metric import get_membership_attack_prob, eval_retain_acc, eval_forget_acc
from unlearn_kd_sample.utils.seed import set_seed

def main():
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--ratio", type=float, default=0.05)
    p.add_argument("--device", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    m = get_model(args.model)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    train, test = get_cifar10()
    retain_subset, unlearn_subset, _ = make_unlearn_retain_split(train, per_class_unlearn_ratio=args.ratio, seed=42)
    retain_loader, unlearn_loader, _ = get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=args.bs)

    mia_rate= get_membership_attack_prob(retain_loader, unlearn_loader, test_loader, m, device)
    r_loss, r_acc = eval_retain_acc(m, retain_loader, device)
    f_loss, f_acc = eval_forget_acc(m, unlearn_loader, device)

    print({
        "retain_acc": r_acc,
        "forget_acc": f_acc,
        "mia_rate_on_unlearn": mia_rate,
    })


if __name__ == "__main__":
    main()
