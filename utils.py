import os
import math
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = ["GPU", "CPU", "Processor", "Normalize", "Writer", "torch_assert"]

GPU = torch.device("cuda")
CPU = torch.device("cpu")


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


class Normalize:
    __mean = torch.tensor([
        [[0.039798092]], [[0.23215543]], [[0.21487042]], [[0.12491968]],
        [[0.23842652]], [[0.061436515]], [[0.024498232]], [[0.2339738]], [[0.014193709]]
    ], dtype=torch.float32, device=GPU)
    __std = torch.tensor([
        [[0.032900818]], [[0.1462142]], [[0.15836369]], [[0.09872178]],
        [[0.14610584]], [[0.056200337]], [[0.015856693]], [[0.14639099]], [[0.008545931]]
    ], dtype=torch.float32, device=GPU)

    @staticmethod
    def do(spec: torch.Tensor) -> torch.Tensor:
        return spec.sub(Normalize.__mean).div(Normalize.__std)

    @staticmethod
    def re(spec: torch.Tensor) -> torch.Tensor:
        return spec.mul(Normalize.__std).add(Normalize.__mean)


class Writer:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._writer: SummaryWriter = SummaryWriter(log_dir)
        self._step_dict: dict[str, int] = {}

    def add_scalar(self, name, value):
        if name not in self._step_dict:
            self._step_dict[name] = 0
        self._writer.add_scalar(name, value, self._step_dict[name])
        self._step_dict[name] += 1


def torch_assert(condition: bool, message: str):
    torch._assert(condition, message)  # noqa


def test():
    spec = torch.randn(9, 600, 800, dtype=torch.float32, device=GPU)
    print(spec)
    no = Normalize.do(spec)
    print(no)
    re = Normalize.re(no)
    print(re)


if __name__ == "__main__":
    test()
