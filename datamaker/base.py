import cv2
import torch
import random
from PIL import Image
import torchvision.transforms.functional as F  # noqa
from torchvision.transforms.transforms import RandomCrop, Resize

__all__ = [
    "DEVICE",
    "Augment", "Overlay", "Target",
    "collate_fn", "imshow"
]

MASK_HOLD = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Augment:
    def __init__(self, mask: torch.Tensor, specs: torch.Tensor):
        self.mask = mask
        self.specs = specs

    @staticmethod
    def _scale(mask: Image, specs: list[Image], _min: float = 0.3, _max: float = 0.7):
        scale = random.uniform(_min, _max)
        new_size = [int(mask.height * scale), int(mask.width * scale)]
        mask = F.resize(mask, new_size)
        specs = [F.resize(band, new_size) for band in specs]
        return mask, specs

    @staticmethod
    def _rotate(mask: Image, specs: list[Image], _min: float = -90, _max: float = 90):
        angle = random.uniform(_min, _max)
        mask = F.rotate(mask, angle)
        specs = [F.rotate(band, angle) for band in specs]
        return mask, specs

    @staticmethod
    def _crop(mask: Image, specs: list[Image], _min: float = 0.6, _max: float = 0.8):
        scale = random.uniform(_min, _max)
        output_size = (int(mask.height * scale), int(mask.width * scale))
        i, j, h, w = RandomCrop.get_params(mask, output_size=output_size)
        mask = F.crop(mask, i, j, h, w)
        specs = [F.crop(band, i, j, h, w) for band in specs]
        return mask, specs

    @staticmethod
    def _v_flip(mask: Image, specs: list[Image]):
        mask = F.vflip(mask)
        specs = [F.vflip(band) for band in specs]
        return mask, specs

    @staticmethod
    def _h_flip(mask: Image, specs: list[Image]):
        mask = F.hflip(mask)
        specs = [F.hflip(band) for band in specs]
        return mask, specs

    @staticmethod
    def _fit(mask: torch.Tensor, specs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonzero_indices = torch.nonzero(mask)
        if nonzero_indices.size(0) == 0:
            return torch.empty(0), torch.empty(0)

        min_row = torch.min(nonzero_indices[:, 0])
        max_row = torch.max(nonzero_indices[:, 0])
        min_col = torch.min(nonzero_indices[:, 1])
        max_col = torch.max(nonzero_indices[:, 1])
        return (
            mask[min_row:max_row + 1, min_col:max_col + 1],
            specs[:, min_row:max_row + 1, min_col:max_col + 1],
        )

    def do(self, p: float = 0.5, hold: float = 0.1) -> bool:
        raw = torch.where(self.mask > MASK_HOLD)[0].size(0)
        mask = F.to_pil_image(self.mask)
        specs = [F.to_pil_image(img) for img in self.specs]
        if random.random() > p:
            mask, specs = self._scale(mask, specs, 0.5, 0.8)
        if random.random() > p:
            mask, specs = self._rotate(mask, specs, -90, 90)
        if random.random() > p:
            mask, specs = self._crop(mask, specs, 0.6, 0.8)
        if random.random() > p:
            mask, specs = self._h_flip(mask, specs)
        if random.random() > p:
            mask, specs = self._v_flip(mask, specs)
        mask = F.to_tensor(mask).squeeze().to(device=DEVICE)
        specs = torch.stack([
            F.to_tensor(band).squeeze() for band in specs  # noqa
        ]).to(device=DEVICE)
        self.mask, self.specs = self._fit(mask, specs)
        res = torch.where(self.mask > MASK_HOLD)[0].size(0)
        return res / raw > hold


class Overlay:
    def __init__(self, back: torch.Tensor):
        self.back = back
        self._big_mask = torch.zeros(back.shape[1:], dtype=torch.float32, device=DEVICE)

    def put(
            self,
            mask: torch.Tensor,
            specs: torch.Tensor,
            x: int,
            y: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = mask.shape
        svi = torch.where(mask > MASK_HOLD)  # small valid index (h, w)
        svi0, svi1 = svi
        bvi0, bvi1 = svi0 + y, svi1 + x
        for i in range(9):
            self.back[i, bvi0, bvi1] = specs[i, svi0, svi1]
        big_mask = self._big_mask.clone()
        big_mask[bvi0, bvi1] = mask[svi0, svi1]
        rect = torch.tensor([x, y, x + w, y + h], dtype=torch.float32, device=DEVICE)
        return big_mask, rect


class Target:
    @staticmethod
    def _resize(img: torch.Tensor, shape: tuple[int, int], limit: tuple[float, float]):
        _, h, w = img.shape
        height_ratio, width_ratio = h / shape[0], w / shape[1]
        ratio_max = max(height_ratio, width_ratio)
        if limit[0] <= ratio_max <= limit[1]:
            return img
        if ratio_max > limit[1]:
            scale = limit[1] / ratio_max
        else:
            scale = limit[0] / ratio_max

        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)
        return Resize((new_h, new_w))(img)

    def __init__(
            self,
            data: dict,
            order: list[int],
            shape: tuple[int, int],
            limit: tuple[float, float]
    ):
        """ 目标类
        :param data: numpy加载的目标原始信息，包含 "spec", "tags"
        :param order: 新通道顺序
        :param shape: 参考尺寸
        :param limit: 相对于参考尺寸的大小范围
        """
        spec = torch.from_numpy(data["spec"]).to(dtype=torch.float32, device=DEVICE)
        spec = self._resize(spec, shape, limit)
        self.mask = spec[0]
        self.spec = spec[1:][order, :, :]
        self.label = torch.tensor(data["tags"][0], dtype=torch.int64, device=DEVICE)
        self.material = torch.tensor(data["tags"][1], dtype=torch.int64, device=DEVICE)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def imshow(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


