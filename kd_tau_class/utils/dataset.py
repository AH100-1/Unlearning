import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def full_forget_retain_loader_train(forget_class, batch_size=256):
    data_root='./data'
    
    # 오타 수정: transfrom -> transform, transfroms -> transforms
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 변수명 수정: full_dataset_trian -> full_dataset_train
    full_dataset_train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    forget_indices = []
    retain_indices = []
    
    # 변수명 수정: full_dataset -> full_dataset_train
    for idx, (_, label) in enumerate(full_dataset_train):
        if label == forget_class:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)  # 수정: forget_indices -> retain_indices
    
    # 변수명 수정
    forget_dataset = Subset(full_dataset_train, forget_indices)
    retain_dataset = Subset(full_dataset_train, retain_indices)  # 수정
    
    # 들여쓰기 수정, 변수명 수정
    full_loader = DataLoader(
        full_dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    forget_loader = DataLoader(
        forget_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    retain_loader = DataLoader(
        retain_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    # 리턴 변수명 수정
    return full_loader, forget_loader, retain_loader

def full_forget_retain_loader_test(forget_class, batch_size=256):
    
    data_root='./data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    full_dataset_test = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    forget_indices = []
    retain_indices = []
    
    for idx, (_, label) in enumerate(full_dataset_test):
        if label == forget_class:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)  # 수정: forget_indices -> retain_indices
    
    forget_dataset_test = Subset(full_dataset_test, forget_indices)
    retain_dataset_test = Subset(full_dataset_test, retain_indices)  # 변수명 수정
    
    # 들여쓰기 수정
    full_loader = DataLoader(
        full_dataset_test, 
        batch_size=batch_size, 
        shuffle=False,  # 테스트는 shuffle=False
        num_workers=2,
        pin_memory=True
    )
    
    forget_loader = DataLoader(
        forget_dataset_test, 
        batch_size=batch_size, 
        shuffle=False,  # 테스트는 shuffle=False
        num_workers=2,
        pin_memory=True
    )
    
    retain_loader = DataLoader(
        retain_dataset_test, 
        batch_size=batch_size, 
        shuffle=False,  # 테스트는 shuffle=False
        num_workers=2,
        pin_memory=True
    )
    
    # 리턴 수정: full_loader 추가
    return full_loader, forget_loader, retain_loader
