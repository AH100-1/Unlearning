import argparse, torch
from unlearn_kd_class.model import get_model
from unlearn_kd_class.dataset import get_cifar10, get_dataloader
from unlearn_kd_class.utils.train import train_teacher
from unlearn_kd_class.utils.seed import set_seed


def main():
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--device", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    m = get_model(args.model)
    train, test = get_cifar10()
    tr_loader = get_dataloader(train, batch_size=args.bs, shuffle=True)
    te_loader = get_dataloader(test, batch_size=256, shuffle=False)

    stats = train_teacher(m, tr_loader, te_loader, device, epochs=args.epochs, save_path=args.save)
    print({"best_acc": stats["best_acc"]})


if __name__ == "__main__":
    main()