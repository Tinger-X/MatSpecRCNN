import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList

__all__ = ["Normalizer"]


class Normalizer(nn.Module):
    def __init__(self, mean: list[float] = None, std: list[float] = None):
        super(Normalizer, self).__init__()
        if mean is None:
            mean = [
                0.039798092, 0.23215543, 0.21487042, 0.12491968,
                0.23842652, 0.061436515, 0.024498232, 0.2339738, 0.014193709
            ]
        if std is None:
            std = [
                0.032900818, 0.1462142, 0.15836369, 0.09872178,
                0.14610584, 0.056200337, 0.015856693, 0.14639099, 0.008545931
            ]
        assert isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std)

        size = len(mean)
        mean = torch.tensor(mean, dtype=torch.float32).view(size, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(size, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images: Tensor) -> ImageList:
        sizes = []
        for i in range(len(images)):
            images[i].sub_(self.mean).div_(self.std)
            sizes.append((images.shape[-2], images.shape[-1]))
        return ImageList(images, sizes)

    def __repr__(self):
        return (
            f"Normalizer(\n"
            f"\tmean={list(self.mean.flatten().cpu().numpy())},\n"
            f"\tstd={list(self.std.flatten().cpu().numpy())}\n)"
        )
