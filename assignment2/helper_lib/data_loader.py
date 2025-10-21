import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Sequence, Optional

def _build_transform(
    in_size: int = 32,
    in_channels: int = 1,
    normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    force_grayscale: bool = True,
):
    ops = []
    if force_grayscale and in_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.Resize((in_size, in_size)))
    ops.append(transforms.ToTensor())
    if normalize_stats is None:
        normalize_stats = ((0.5,) * in_channels, (0.5,) * in_channels)
    mean, std = normalize_stats
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)

def get_data_loader(
    data_dir: str,
    batch_size: int = 32,
    train: bool = True,
    num_workers: int = 2,
    in_size: int = 32,
    in_channels: int = 1,
    normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
):
    tfms = _build_transform(in_size, in_channels, normalize_stats, force_grayscale=(in_channels == 1))
    dataset = datasets.ImageFolder(root=data_dir, transform=tfms)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
