from torch.utils.data import DataLoader
import torch


def collate_with_flag(batch):
    # print("[DEBUG collate_with_flag] first sample len:", len(batch[0]))
    first = batch[0]
    if len(first) == 3:  # (x,y,flag)
        xs, ys, fs = zip(*batch)
        return torch.stack(xs), torch.tensor(ys), torch.tensor(fs)
    elif len(first) == 2:  # (x,y)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys)
    else:
        raise ValueError("Unexpected batch format")

def get_dataloader(dataset, batch_size=128, num_workers=4,
                   shuffle=True, drop_last=False, with_flag=False):
    
    if with_flag:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_with_flag   
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
