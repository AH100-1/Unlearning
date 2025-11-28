# train_test_acc.py  (수정판)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, dataset, datatype, epochs=30, lr=1e-4, batch_size=256, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.1,           # 더 크게 시작
                            momentum=0.9,
                            weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    if len(next(iter(dataloader)))==3:
        for epoch in range(epochs):
            running, correct, total = 0.0, 0, 0
            for images, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step()

                running += loss.item()
                pred = outputs.argmax(1)
                total += labels.size(0); correct += (pred == labels).sum().item()

            scheduler.step()
            print(f'[Train:{datatype}] Epoch {epoch+1}/{epochs} '
                  f'Loss {running/len(dataloader):.4f} Acc {100*correct/total:.2f}%')
    
    else:
        for epoch in range(epochs):
            running, correct, total = 0.0, 0, 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step()

                running += loss.item()
                pred = outputs.argmax(1)
                total += labels.size(0); correct += (pred == labels).sum().item()

            scheduler.step()
            print(f'[Train:{datatype}] Epoch {epoch+1}/{epochs} '
                  f'Loss {running/len(dataloader):.4f} Acc {100*correct/total:.2f}%')
        
    return model

@torch.no_grad()
def model_test(model, dataset, data_type, batch_size=256, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    correct, total = 0, 0
    
    if len(next(iter(loader)))==3:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            total += labels.size(0); correct += (pred == labels).sum().item()
    else:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            total += labels.size(0); correct += (pred == labels).sum().item()
        

    acc = 100 * correct / total
    print(f'[Test:{data_type}] Acc {acc:.2f}%')
    return acc
