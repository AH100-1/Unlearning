from torch.utils.data import ConcatDataset, Dataset
from .get_dataloader import get_dataloader

class FlagConcatDataset(Dataset):
    """ConcatDataset but preserves (x, y, flag)."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for d in datasets:
            total += len(d)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        for i, csize in enumerate(self.cumulative_sizes):
            if idx < csize:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_sizes[i-1]]


def get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=128, num_workers=4):
    retain_loader = get_dataloader(retain_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True, with_flag=True)
    unlearn_loader = get_dataloader(unlearn_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True, with_flag=True)
    joint_loader = get_dataloader(FlagConcatDataset([retain_subset, unlearn_subset]), batch_size=batch_size, num_workers=num_workers, shuffle=True, with_flag=True)
    return retain_loader, unlearn_loader, joint_loader