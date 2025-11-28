from typing import Tuple
import torchvision.transforms as T
from torchvision import datasets


def get_cifar10(data_root: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    # standard augmentation for train; eval transforms for test
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    return train, test