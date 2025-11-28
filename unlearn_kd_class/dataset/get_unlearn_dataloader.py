from torch.utils.data import ConcatDataset, Dataset
from .get_dataloader import get_dataloader


def get_unlearn_dataloaders(retain_subset, unlearn_subset, batch_size=128, num_workers=4):
    retain_loader = get_dataloader(retain_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    unlearn_loader = get_dataloader(unlearn_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return retain_loader, unlearn_loader