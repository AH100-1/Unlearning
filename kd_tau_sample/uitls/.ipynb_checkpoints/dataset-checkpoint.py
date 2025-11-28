import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np


from torch.utils.data import Dataset

class UnlearnFullTrain(Dataset):
    """forget/retain 두 Subset을 합쳐서 (img, label, check) 반환.
       check: forget=1, retain=0
    """
    def __init__(self, forget_set, retain_set):
        super().__init__()
        self.forget_set = forget_set
        self.retain_set = retain_set

        # (출처, 하위 인덱스)
        self.samples = [("forget", i) for i in range(len(self.forget_set))]
        self.samples += [("retain", i) for i in range(len(self.retain_set))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, j = self.samples[idx]
        if src == "forget":
            img, label = self.forget_set[j]
            check = 1
        else:
            img, label = self.retain_set[j]
            check = 0
        return img, label, check





def split_label(full_data, ratio=0.1, seed=42, num_classes=10):
    """클래스별로 ratio 비율만큼 forget으로 보냄. 나머지는 retain."""
    labels = full_data.targets if hasattr(full_data, "targets") \
             else [full_data[i][1] for i in range(len(full_data))]

    buckets = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels):
        buckets[int(y)].append(idx)

    rng = np.random.default_rng(seed)
    forget_idx, retain_idx = [], []

    for idxs in buckets:
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n_forget = int(len(idxs) * ratio)
        forget_idx.extend(idxs[:n_forget].tolist())
        retain_idx.extend(idxs[n_forget:].tolist())

    retain_set = Subset(full_data, retain_idx)
    forget_set = Subset(full_data, forget_idx)
    return retain_set, forget_set  # ← 순서 고정: (retain, forget)





def full_forget_retain_loader_train(forget_ratio=0.1, batch_size=256, seed=42):
    data_root = './data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    full_dataset_train = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )

    retain_dataset, forget_dataset = split_label(
        full_dataset_train, ratio=forget_ratio, seed=seed, num_classes=10
    )
    
    combined = UnlearnFullTrain(forget_set=forget_dataset, retain_set=retain_dataset)
   

    full_loader   = DataLoader(full_dataset_train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    forget_loader = DataLoader(forget_dataset,   batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    retain_loader = DataLoader(retain_dataset,   batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    combined_loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    
    return full_loader, combined_loader, forget_loader, retain_loader

def full_forget_retain_loader_test(forget_ratio=0.1, batch_size=256, seed=42):
    """보통 test는 full만 쓰지만, 요청하신 스타일대로 split도 리턴."""
    data_root = './data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    full_dataset_test = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    retain_dataset_test, forget_dataset_test = split_label(
        full_dataset_test, ratio=forget_ratio, seed=seed, num_classes=10
    )
    
    combined = UnlearnFullTrain(forget_set=forget_dataset_test, retain_set=retain_dataset_test)
    
    full_loader   = DataLoader(full_dataset_test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    forget_loader = DataLoader(forget_dataset_test,batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    retain_loader = DataLoader(retain_dataset_test,batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    combined_loader = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return full_loader, combined_loader, forget_loader, retain_loader

