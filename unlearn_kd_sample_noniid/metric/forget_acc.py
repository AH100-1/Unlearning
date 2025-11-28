from typing import Tuple
import torch


@torch.no_grad()
def eval_forget_acc(model, unlearn_loader, device) -> Tuple[float, float]:
    """
    Accuracy on UNLEARN set (lower is better if true forgetting is desired).
    Returns (loss, acc).
    """
    model.eval().to(device)
    correct, total = 0, 0
    import torch.nn.functional as F
    loss_sum = 0.0
    for x, y, _flag in unlearn_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)