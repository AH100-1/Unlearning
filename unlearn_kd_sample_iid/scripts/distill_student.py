import argparse, torch
from unlearn_kd_sample.model import get_model
from unlearn_kd_sample.dataset import get_cifar10, get_dataloader
from unlearn_kd_sample.utils.distiller import distill, KDConfig
from unlearn_kd_sample.utils.seed import set_seed


def main():
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, required=True)
    p.add_argument("--student", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--device", type=int, default=3)
    p.add_argument("--save", type=str, required=True)
    args = p.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    t = get_model(args.teacher)
    s = get_model(args.student)
    t.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu"))

    train, test = get_cifar10()
    tr = get_dataloader(train, batch_size=args.bs)
    te = get_dataloader(test, batch_size=256, shuffle=False)

    stats = distill(s, t, tr, te, device, epochs=args.epochs, cfg=KDConfig(T=args.T, alpha=args.alpha), save=args.save)
    print({"best_acc": stats["best_acc"]})


if __name__ == "__main__":
    main()