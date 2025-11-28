from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from .test import test


def train_teacher(model, train_loader, test_loader, device, epochs=100, lr=0.1, wd=5e-4, momentum=0.9, save_path: str = None) -> Dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    best_acc = 0.0
    best_state = None
    batch = next(iter(train_loader))
    for ep in range(1, epochs + 1):
        model.train()
        
        if len(batch) == 2:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        elif len(batch) == 3:
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        else:
            raise ValueError("Unsupported batch format")

        _, acc = test(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
        # scheduler.step()

    # load best back (for immediate use)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_acc": best_acc}