from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Dataset,Subset

class WithFlagDataset(Dataset):
    """Dataset wrapper that exposes a per-sample flag (retain=0 / unlearn=1)."""
    def __init__(self, dataset, indices: List[int], flag_value: int):
        self.dataset = dataset
        self.indices = indices
        self.flag_value = flag_value

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]          # ✅ 원본 dataset 인덱스
        x, y = self.dataset[real_idx]         # ✅ CIFAR-10에서 (x,y) 꺼냄
        return x, y, self.flag_value          # ✅ flag 붙여서 반환

def make_unlearn_retain_split(train_dataset, per_class_unlearn_ratio: float = 0.1,
                              seed: int = 42) -> Tuple[WithFlagDataset, WithFlagDataset, Dict[int, Tuple[List[int], List[int]]]]:
    """
    Split CIFAR-10 train set per class so that exactly ratio of samples per class
    are assigned to UNLEARN set.
    Returns (retain_dataset, unlearn_dataset, index_map_by_class).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(train_dataset.targets)
    retain_indices: List[int] = []
    unlearn_indices: List[int] = []
    idx_map: Dict[int, Tuple[List[int], List[int]]] = {}

    for c in range(10):
        if c<=5:
            cls_idx = np.where(targets == c)[0]
            rng.shuffle(cls_idx)
            n_unl = max(1, int(round(len(cls_idx) * per_class_unlearn_ratio)))
            u_idx = cls_idx[:n_unl].tolist()
            r_idx = cls_idx[n_unl:].tolist()
            unlearn_indices.extend(u_idx)
            retain_indices.extend(r_idx)
            idx_map[c] = (r_idx, u_idx)
        else:
            cls_idx = np.where(targets == c)[0]
            rng.shuffle(cls_idx)
            r_idx = cls_idx.tolist()
            retain_indices.extend(r_idx)
            idx_map[c] = (r_idx, [])    

    retain_dataset = WithFlagDataset(train_dataset, retain_indices, flag_value=0)
    unlearn_dataset = WithFlagDataset(train_dataset, unlearn_indices, flag_value=1)
    return retain_dataset, unlearn_dataset, idx_map
