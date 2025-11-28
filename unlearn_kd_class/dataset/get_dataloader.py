from torch.utils.data import DataLoader
import torch


def get_dataloader(dataset, batch_size=128, num_workers=4,
                   shuffle=True, drop_last=False):
    
    
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
    )
 
