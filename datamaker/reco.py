# 目标检测数据集生成器
import os
import cv2
import math
import torch
import random
import numpy as np
from typing import Literal
from torchvision.transforms.transforms import Resize
from torch.utils.data import Dataset as TorchDataset, DataLoader

from datamaker.base import DEVICE, Augment, Overlay, Target, collate_fn, imshow

__all__ = ["RecoMaker", "RecoDataset"]


class PositionAssigner:
    def __init__(self, size: torch.Size):
        """ 位置分配器
        :param size: (h, w) 背景图大小
        """
        self._size = size
        self._rects = []

    @staticmethod
    def _one_lapped(rect1, rect2):
        """分离轴定理检测重叠"""
        return not (
                rect1[0] + rect1[2] <= rect2[0] or
                rect1[0] >= rect2[0] + rect2[2] or
                rect1[1] + rect1[3] <= rect2[1] or
                rect1[1] >= rect2[1] + rect2[3]
        )

    def _overlap(self, rect):
        for placed in self._rects:
            if self._one_lapped(rect, placed):
                return True
        return False

    def add_rect(self, size: torch.Size, max_retry: int = 10):
        """ 添加一个目标
        :param size: (h, w) 目标大小
        :param max_retry: 最大重试次数
        :return: None or (x, y)
        """
        positions = [
            (random.randint(0, self._size[1] - size[1]), random.randint(0, self._size[0] - size[0]))
            for _ in range(max_retry)
        ]
        for x, y in positions:
            rect = (x, y, size[1], size[0])
            if not self._overlap(rect):
                self._rects.append(rect)
                return x, y
        return None


class RecoDataset(TorchDataset):
    def __init__(self, bgs: list, datas: list[Target], size: int, seed: int, idx: list[int]):
        self._seed, self._size = seed, size
        self._bgs = bgs
        self._datas = datas
        self._bg_idx = list(range(len(bgs)))
        self._tar_idx = list(range(len(datas)))
        self._idx = idx
        self.reset()

    def reset(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        back = self._bgs[random.choice(self._bg_idx)].clone()
        ov = Overlay(back)
        pa = PositionAssigner(back.shape[1:])
        idx = 0
        if random.random() < 0.05:
            return back[self._idx, :, :], {
                "masks": torch.zeros((0, *back.shape[1:]), dtype=torch.float32, device=DEVICE),
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=DEVICE),
                "labels": torch.zeros(0, dtype=torch.int64, device=DEVICE),
                "materials": torch.zeros(0, dtype=torch.int64, device=DEVICE)
            }
        size = random.randint(1, 4)
        masks = torch.zeros((size, *back.shape[1:]), dtype=torch.float32, device=DEVICE)
        bboxes = torch.zeros((size, 4), dtype=torch.float32, device=DEVICE)
        labels = torch.zeros((size,), dtype=torch.int64, device=DEVICE)
        materials = torch.zeros((size,), dtype=torch.int64, device=DEVICE)
        while idx < size:
            data = self._datas[random.choice(self._tar_idx)]
            aug = Augment(data.mask.clone(), data.spec.clone())
            if not aug.do():
                continue
            pos = pa.add_rect(aug.mask.shape)
            if not pos:
                continue
            mask, bbox = ov.put(aug.mask, aug.specs, pos[0], pos[1])
            masks[idx] = mask
            bboxes[idx] = bbox
            labels[idx] = data.label
            materials[idx] = data.material
            idx += 1
        target = {"masks": masks, "boxes": bboxes, "labels": labels, "materials": materials}
        return ov.back[self._idx, :, :], target


class RecoMaker:
    def __init__(
            self,
            bg_path: str,
            tar_path: str,
            data_type: Literal["rgb", "540", "full"] = "full",
            bg_shape: tuple[int, int] = (600, 800),
            tar_range: tuple[float, float] = (0.4, 0.6)
    ):
        """
        :param bg_path: 背景文件夹路径
        :param tar_path: 目标文件夹路径
        :param data_type: 数据类型，rgb、540、全通道
        :param bg_shape: (h, w) 图像大小，默认：(600, 800)
        :param tar_range: 目标相对于图像的大小范围，默认：(0.4, 0.6)

           i:       0       1       2       3       4       5      6       7       8
        原始顺序：[450(0), 540(1), 590(2), 600(3), 650(4), 690(5),  R(6) , G(7) ,  B(8)]
        更新顺寻：[450(0),  B(8) , 540(1), 590(2),  G(7) , 600(3), 650(4), R(6), 690(5)]
        """
        order = [0, 8, 1, 2, 7, 3, 4, 6, 5]
        self.rgb_order = [7, 4, 1]
        assert data_type in ["rgb", "540", "full"]
        self._idx = self.rgb_order
        if data_type == "540":
            self._idx = [2]
        elif data_type == "full":
            self._idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        resize = Resize(bg_shape)

        self._bgs = [
            resize(torch.tensor(
                np.load(f"{bg_path}/{name}")["images"][order, :, :],
                dtype=torch.float32,
                device=DEVICE
            ))
            for name in os.listdir(bg_path)
        ]
        self._datas = [
            Target(np.load(f"{tar_path}/{name}"), order, bg_shape, tar_range)
            for name in os.listdir(tar_path)
        ]

    def any(self, batch: int, size: int, seed: int) -> tuple[DataLoader, callable]:
        dataset = RecoDataset(self._bgs, self._datas, size, seed, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def train(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = RecoDataset(self._bgs, self._datas, 2 ** 12, 19, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def test(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = RecoDataset(self._bgs, self._datas, 2 ** 10, 29, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def val(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = RecoDataset(self._bgs, self._datas, 2 ** 10, 59, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset


class Processor:
    def __init__(self, total: int, length: int = 50, prefix: str = None):
        self._cur = 1
        self._tar = total
        self._len = length
        self._pre = prefix or "Progress"
        self._ss = ("=", ">", " ")

    def _calc_ss(self, per: float):
        num1 = self._len * per
        if num1.is_integer():
            num1 = int(num1)
            num3 = self._len - num1
            return self._ss[0] * num1, "", self._ss[2] * num3
        num1 = math.floor(num1)
        return self._ss[0] * num1, self._ss[1], self._ss[2] * (self._len - 1 - num1)

    def next(self, **kwargs):
        per = self._cur / self._tar
        inner = "".join(self._calc_ss(per))
        info = ""
        if kwargs is not None:
            info = "".join([f", {k}={v}" for k, v in kwargs.items()])

        print(
            f"\r{self._pre}: {self._cur:03d}/{self._tar:03d}, "
            f"[{inner}], {per * 100:6.2f}%{info}",
            end=""
        )
        self._cur = min(self._cur + 1, self._tar)

    def done(self, **kwargs):
        self._cur = self._tar
        self.next(**kwargs)
        print()


def calc_mean_std():
    import time

    size, batch = 2 ** 16, 2 ** 6
    datamaker = RecoMaker("../data/bg", "../data/tar", bg_shape=(384, 512))
    loader, _ = datamaker.any(batch, size, 73)  # 只能使用 any 返回未归一化的数据用于计算 mean 和 std
    print(f"开始计算，数据量：", size)
    mean = torch.zeros(9, dtype=torch.float32, device=DEVICE)
    std = torch.zeros(9, dtype=torch.float32, device=DEVICE)
    process = Processor(size // batch)
    start = time.time()
    for images, _ in loader:
        mean_batch = torch.mean(images, dim=(2, 3))
        std_batch = torch.std(images, dim=(2, 3))
        mean.add_(torch.sum(mean_batch, dim=0))
        std.add_(torch.sum(std_batch, dim=0))
        process.next(cost=f"{time.time() - start:.2f}s")
    mean.div_(size)
    std.div_(size)
    """
    mean = [
        0.039798092, 0.23215543, 0.21487042, 0.12491968,
        0.23842652, 0.061436515, 0.024498232, 0.2339738, 0.014193709
    ]
    std = [
        0.032900818, 0.1462142, 0.15836369, 0.09872178,
        0.14610584, 0.056200337, 0.015856693, 0.14639099, 0.008545931
    ]
    """
    print(f"计算完成，耗时：{time.time() - start:.2f}秒，计算结果：")
    print("image_mean =", list(mean.cpu().detach().numpy()))
    print("image_std =", list(std.cpu().detach().numpy()))


def common(image: np.ndarray, target: dict[str, torch.Tensor]):
    imshow(image)
    masks = target["masks"].cpu().detach().numpy()
    boxes = target["boxes"].cpu().detach().numpy().astype(int)
    labels = target["labels"].cpu().detach().numpy()
    materials = target["materials"].cpu().detach().numpy()
    if len(masks) == 0:
        print("no targets in image")
        return
    for i in range(len(masks)):
        print(f"box: {boxes[i]}, label: {labels[i]}, material: {materials[i]}")
        box = boxes[i]
        img = cv2.rectangle(masks[i], (box[0], box[1]), (box[2], box[3]), 1, 1)
        imshow(img)


def test_rgb():
    datamaker = RecoMaker(
        "../data/bg", "../data/tar", data_type="rgb",
        bg_shape=(384, 512), tar_range=(0.4, 0.9)
    )
    loader, reset_fn = datamaker.train(16)
    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy()
            common(image[::-1, :, :].transpose((1, 2, 0)), target)
        break


def test_540():
    datamaker = RecoMaker(
        "../data/bg", "../data/tar", data_type="540",
        bg_shape=(384, 512), tar_range=(0.4, 0.9)
    )
    loader, reset_fn = datamaker.train(16)
    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy()
            common(image.transpose((1, 2, 0)), target)
        break


def test_full():
    datamaker = RecoMaker(
        "../data/bg", "../data/tar", data_type="full",
        bg_shape=(384, 512), tar_range=(0.4, 0.9)
    )
    loader, reset_fn = datamaker.train(16)
    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy()
            common(image[datamaker.rgb_order[::-1]].transpose((1, 2, 0)), target)
        break


if __name__ == "__main__":
    # calc_mean_std()
    test_rgb()
