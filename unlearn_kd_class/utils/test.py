from typing import Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def test(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    model.to(device)
    correct, total = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)