# 目标跟踪数据集生成器
import os
import cv2
import torch
import random
import numpy as np
from typing import Literal
from torchvision.transforms.transforms import Resize
from torch.utils.data import Dataset as TorchDataset, DataLoader

from datamaker.base import DEVICE, Augment, Overlay, Target, collate_fn

__all__ = ["TrackMaker", "TrackDataset"]


class MotionGenerator:
    """轨迹生成器，支持18种运动算子模拟追踪态势演变"""

    def __init__(self, w_eff: int, h_eff: int, seq_len: int):
        self.we = max(w_eff, 1)  # 有效横向移动空间 (W - target_w)
        self.he = max(h_eff, 1)  # 有效纵向移动空间 (H - target_h)
        self.t = np.linspace(0, 1, seq_len)

        # 边界与锚点参数预设
        self.ox, self.oy = self.we / 2, self.he / 2
        self.rx, self.ry = self.we * 0.4, self.he * 0.4
        self.wx, self.hy = self.we * 0.8, self.he * 0.8
        self.ox_rect, self.oy_rect = self.we * 0.1, self.he * 0.1

    def _rect_map(self, s):
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        for i, val in enumerate(s):
            val = val % 4
            if val < 1:
                x[i], y[i] = self.ox_rect + self.wx * val, self.oy_rect
            elif val < 2:
                x[i], y[i] = self.ox_rect + self.wx, self.oy_rect + self.hy * (val - 1)
            elif val < 3:
                x[i], y[i] = self.ox_rect + self.wx * (3 - val), self.oy_rect + self.hy
            else:
                x[i], y[i] = self.ox_rect, self.oy_rect + self.hy * (4 - val)
        return x, y

    def get_motion(self, type_idx: int) -> tuple[np.ndarray, np.ndarray]:
        t = self.t
        rand_y = random.uniform(0, 1) * self.he
        rand_x = random.uniform(0, 1) * self.we

        if type_idx == 1:  # 1A: 匀速直线
            return self.we * t, self.he * t
        elif type_idx == 2:  # 1B: 二次变速直线
            return self.we * (t ** 2), self.he * (t ** 2)
        elif type_idx == 3:  # 1C: 开方减速直线
            return self.we * np.sqrt(t), self.he * np.sqrt(t)
        elif type_idx == 4:  # 2A: 水平三角往返
            return self.we * (1 - 2 * np.abs(t - 0.5)), np.full_like(t, rand_y)
        elif type_idx == 5:  # 2B: 水平二次变速往返
            return self.we * 4 * t * (1 - t), np.full_like(t, rand_y)
        elif type_idx == 6:  # 2C: 水平正弦往返
            return self.we * (1 + np.sin(np.pi * (2 * t - 1))) / 2, np.full_like(t, rand_y)
        elif type_idx == 7:  # 3A: 垂直三角往返
            return np.full_like(t, rand_x), self.he * (1 - 2 * np.abs(t - 0.5))
        elif type_idx == 8:  # 3B: 垂直二次变速往返
            return np.full_like(t, rand_x), self.he * 4 * t * (1 - t)
        elif type_idx == 9:  # 3C: 垂直正弦往返
            return np.full_like(t, rand_x), self.he * (1 + np.sin(np.pi * (2 * t - 1))) / 2
        elif type_idx == 10:  # 4A: 矩形匀速绕圈
            return self._rect_map(4 * t)
        elif type_idx == 11:  # 4B: 矩形各边二次变速
            return self._rect_map(4 * (t ** 2))
        elif type_idx == 12:  # 4C: 矩形各边正弦变速
            return self._rect_map(4 * (1 - np.cos(np.pi * t)) / 2)
        elif type_idx == 13:  # 5A: 匀速圆周
            return self.ox + self.rx * np.cos(2 * np.pi * t), self.oy + self.ry * np.sin(2 * np.pi * t)
        elif type_idx == 14:  # 5B: 圆周二次变速
            return self.ox + self.rx * np.cos(2 * np.pi * (t ** 2)), self.oy + self.ry * np.sin(2 * np.pi * (t ** 2))
        elif type_idx == 15:  # 5C: 圆周正弦变速
            theta = np.pi * (1 - np.cos(np.pi * t))
            return self.ox + self.rx * np.cos(theta), self.oy + self.ry * np.sin(theta)
        elif type_idx == 16:  # 6A: 匀速李萨如(8字)
            return self.ox + self.rx * np.sin(2 * np.pi * t), self.oy + self.ry * np.sin(4 * np.pi * t)
        elif type_idx == 17:  # 6B: 频率二次变速8字
            return self.ox + self.rx * np.sin(2 * np.pi * (t ** 2)), self.oy + self.ry * np.sin(4 * np.pi * (t ** 2))
        elif type_idx == 18:  # 6C: 调制变速8字
            mod = 1 + 0.5 * np.sin(2 * np.pi * t)
            return self.ox + self.rx * np.sin(2 * np.pi * t * mod), self.oy + self.ry * np.sin(4 * np.pi * t * mod)
        else:
            return self.we * t, self.he * t


class TrackDataset(TorchDataset):
    def __init__(self, bgs: list, datas: list[Target], size: int, seq_len: int, seed: int, idx: list[int]):
        self._seed, self._size = seed, size
        self.seq_len = seq_len
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
        # 选取基础背景和单个目标(Tracking任务通常围绕单目标时序展开)
        back_base = self._bgs[random.choice(self._bg_idx)].clone()
        data = self._datas[random.choice(self._tar_idx)]

        # 序列级别仅做一次基础形态增强，以维持追踪过程中的形态一致性/自然过渡
        aug = Augment(data.mask.clone(), data.spec.clone())
        while not aug.do(p=0.4):  # 确保生成有效增强
            aug = Augment(data.mask.clone(), data.spec.clone())

        th, tw = aug.mask.shape
        w_eff = back_base.shape[2] - tw
        h_eff = back_base.shape[1] - th

        # 实例化轨迹生成器，随机选取1-18类运动模式
        mg = MotionGenerator(w_eff, h_eff, self.seq_len)
        motion_idx = random.randint(1, 18)
        xs, ys = mg.get_motion(motion_idx)

        seq_images = []
        seq_masks = []
        seq_bboxes = []
        seq_labels = []
        seq_materials = []

        # 逐帧进行合并
        for i in range(self.seq_len):
            frame_back = back_base.clone()
            ov = Overlay(frame_back)
            x, y = int(xs[i]), int(ys[i])

            # 安全防越界保护
            x = max(0, min(x, w_eff))
            y = max(0, min(y, h_eff))

            mask, bbox = ov.put(aug.mask, aug.specs, x, y)
            seq_images.append(ov.back[self._idx, :, :])
            seq_masks.append(mask)
            seq_bboxes.append(bbox)
            seq_labels.append(data.label)
            seq_materials.append(data.material)

        target = {
            "masks": torch.stack(seq_masks),
            "boxes": torch.stack(seq_bboxes),
            "labels": torch.stack(seq_labels),
            "materials": torch.stack(seq_materials)
        }
        return torch.stack(seq_images), target


class TrackMaker:
    def __init__(
            self,
            bg_path: str,
            tar_path: str,
            data_type: Literal["rgb", "540", "full"] = "full",
            seq_len: int = 50,
            bg_shape: tuple[int, int] = (600, 800),
            tar_range: tuple[float, float] = (0.1, 0.3)  # 目标适当调小以容纳运动范围
    ):
        order = [0, 8, 1, 2, 7, 3, 4, 6, 5]
        self.rgb_order = [7, 4, 1]
        assert data_type in ["rgb", "540", "full"]
        self._idx = self.rgb_order
        if data_type == "540":
            self._idx = [2]
        elif data_type == "full":
            self._idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.seq_len = seq_len
        resize = Resize(bg_shape)

        self._bgs = [
            resize(torch.tensor(
                np.load(f"{bg_path}/{name}")["images"][order, :, :],
                dtype=torch.float32,
                device=DEVICE
            ))
            for name in os.listdir(bg_path) if name.endswith(".npz")
        ]
        self._datas = [
            Target(np.load(f"{tar_path}/{name}"), order, bg_shape, tar_range)
            for name in os.listdir(tar_path) if name.endswith(".npz")
        ]

    def any(self, batch: int, size: int, seed: int) -> tuple[DataLoader, callable]:
        dataset = TrackDataset(self._bgs, self._datas, size, self.seq_len, seed, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def train(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = TrackDataset(self._bgs, self._datas, 2 ** 10, self.seq_len, 19, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def test(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = TrackDataset(self._bgs, self._datas, 2 ** 8, self.seq_len, 29, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset

    def val(self, batch: int) -> tuple[DataLoader, callable]:
        dataset = TrackDataset(self._bgs, self._datas, 2 ** 8, self.seq_len, 59, self._idx)
        return DataLoader(dataset, batch_size=batch, collate_fn=collate_fn), dataset.reset


def common_track(images: np.ndarray, target: dict[str, torch.Tensor], delay=40):
    """序列化追踪展示函数"""
    seq_len = images.shape[0]
    boxes = target["boxes"].cpu().detach().numpy().astype(int)
    labels = target["labels"].cpu().detach().numpy()

    for t in range(seq_len):
        img = images[t].copy()
        box = boxes[t]

        if len(box) == 4:
            # 绘制绿色边界框追踪
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # 添加帧率与标签显示
            cv2.putText(
                img, f"Frame: {t + 1} | Label: {labels[t]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        cv2.imshow("Tracking Sequence Visualization", img)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def test_full():
    trackmaker = TrackMaker(
        "../data/bg", "../data/tar", data_type="full",
        seq_len=60,  # 每个序列长度为60帧
        bg_shape=(384, 512),
        tar_range=(0.1, 0.3)
    )
    loader, reset_fn = trackmaker.train(batch=2)

    for images, targets in loader:
        # 遍历Batch中的序列
        for seq_idx in range(len(images)):
            # 取出单个序列，将其形状转换为可以被OpenCV显示的 numpy: [seq_len, H, W, C]
            seq_image = images[seq_idx].cpu().detach().numpy()
            # 采用 RGB 频段展示：取出对应的 3 个通道并在最后维度转置
            seq_rgb = seq_image[:, trackmaker.rgb_order[::-1]].transpose((0, 2, 3, 1))
            seq_target = targets[seq_idx]

            common_track(seq_rgb, seq_target)
        break  # 仅展示第一个batch


if __name__ == "__main__":
    test_full()
