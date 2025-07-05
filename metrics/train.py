"""
usage:
```bash
    # run in NVIDIA Tesla T4
    export PYTHONPATH=/path/to/MatSpecRCNN
    cd /path/to/MatSpecRCNN/metrics
    # MaskRCNN-5 (current)
    python -u train.py -m mask-rcnn -e 10 -b 4 > ../logs/mask-rcnn/train.log 2>&1 &
    # MaskRCNN-3 (old)
    python -u train.py -m mask-rcnn -e 10 -8 4 > ../logs/mask-rcnn-3/train.log 2>&1 &
    # MatSpecRCNN-RGB
    python -u train.py -m self-rgb -e 10 -b 8 > ../tmp/self-rgb/train.log 2>&1 &
    # MatSpecRCNN-540
    python -u train.py -m self-540 -e 10 -b 8 > ../tmp/self-540/train.log 2>&1 &
```
"""

import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader

import utils
import models
from datamaker import Datamaker


def train(
        model,
        train_handler: tuple[DataLoader, callable],
        filename: str,
        epoches: int,
):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # writer = utils.Writer(f"../logs/{filename}")

    # 训练循环
    model.train()
    count, best = len(train_handler[0]), float("inf")
    for epoch in range(epoches):
        epoch_str = f"{epoch + 1:02d}/{epoches:2d}"
        train_handler[1]()

        epoch_loss = 0.0
        process = utils.Processor(total=count, prefix=f"Train[{epoch_str}]")
        for images, targets in train_handler[0]:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            process.next(loss=f"{losses:.4f}", total=f"{epoch_loss:.4f}")
            # writer.add_scalar("Loss.Detail", losses)
        epoch_loss /= count
        process.done(loss=f"{epoch_loss:.4f}")
        # writer.add_scalar("Loss.Summary", epoch_loss)
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), f"../weights/{filename}.pt")


def args_parse():
    parser = argparse.ArgumentParser("Metrics Model Training")
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        choices=["mask-rcnn", "self-rgb", "self-540"], help="Model name"
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=10, help="Training epoches"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=8, help="Training batch size"
    )
    args = parser.parse_args()
    print(
        f"[Arguments] => model: {args.model}, epoch: {args.epoch}, batch: {args.batch}\n"
    )
    return args


def main():
    args = args_parse()
    if args.model == "mask-rcnn":
        model_args = (7,)
        model_cls = models.MaskRCNN_ResNet50
        data_type = "rgb"
    elif args.model == "self-rgb":
        model_args = (7, 5)
        model_cls = models.MatSpecRCNN_RGB
        data_type = "rgb"
    else:
        model_cls = models.MatSpecRCNN_540
        model_args = (7, 5)
        data_type = "540"
    datamaker = Datamaker(
        "../images/bg", "../images/tar", data_type=data_type,  # noqa
        bg_shape=(384, 512), tar_range=(0.4, 0.8)
    )
    model = model_cls(*model_args)
    model.to(utils.GPU)
    train_handler = datamaker.train(batch=args.batch)
    train(model, train_handler, filename=args.model, epoches=args.epoch)


if __name__ == "__main__":
    main()
