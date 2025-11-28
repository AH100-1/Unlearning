from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import Dataset, Subset

def make_unlearn_retain_split(train_dataset, class_target:int = 0,
                              seed: int = 42) :
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

    for idx, (img, label) in enumerate(train_dataset):
        if label == class_target:
            unlearn_indices.append(idx)
        else:
            retain_indices.append(idx)

        idx_map.setdefault(label, ([], []))
        if label == class_target:
            idx_map[label][1].append(idx)  # unlearn indices for this class
        else:
            idx_map[label][0].append(idx)  # retain indices for this class      

    retain_dataset = Subset(train_dataset, retain_indices)
    unlearn_dataset = Subset(train_dataset, unlearn_indices)
    return retain_dataset, unlearn_dataset
