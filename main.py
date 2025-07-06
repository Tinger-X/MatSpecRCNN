"""
usage:
```bash
    # run in NVIDIA Tesla T4
    cd /path/to/MatSpecRCNN
    # warm-up train
    python -u main.py train --epoch 10 --batch 8 --save warm-up > logs/warm-up/train.log 2>&1 &
    # fine-turn train
    python -u main.py train --epoch 20 --batch 8 --load warm-up --save self-full > logs/self-full/train.log 2>&1 &
    # show model
    python main.py show > logs/self-full/model.txt
    # inference (run in RTX 3060)
    python -u main.py infer --epoch 20 --load self-full --batch 4
```
"""

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

import utils
from model.network import MatSpecRCNN
from datamaker import Datamaker


def evaluate(
        model: MatSpecRCNN,
        data_handler: tuple[DataLoader, callable],
        writer: utils.Writer,
        prefix: str = "Eval: "
):
    total_loss = 0.0
    count = len(data_handler[0])
    process = utils.Processor(total=count, prefix=prefix)

    data_handler[1]()
    with torch.no_grad():  # 禁用梯度计算
        for images, targets in data_handler[0]:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())  # 总损失
            total_loss += losses.item()  # noqa
            process.next(loss=f"{losses:.4f}", total=f"{total_loss:.4f}")
            writer.add_scalar("Eval Loss.Detail", losses)

    total_loss /= count
    process.done(loss=f"{total_loss:.4f}")
    writer.add_scalar("Eval Loss.Summary", total_loss)
    return total_loss


def inference(
        model: MatSpecRCNN,
        data_handler: tuple[DataLoader, callable],
        rgb_order: list[int]
):
    def contrast(color):
        return 255 - color[0], 255 - color[1], 255 - color[2]

    model.eval()
    # ["Unknown", "Tank_48-01", "Tank_81", "Tank_T99", "Air_102", "South_215", "Air_Wright"]
    classes = [
        ("Unknown", (0, 0, 255)), ("Tank_48-01", (20, 20, 255)),
        ("Tank_81", (30, 30, 255)), ("Tank_T99", (40, 40, 255)),
        ("Air_102", (20, 255, 20)), ("South_215", (30, 255, 30)),
        ("Air_Wright", (40, 255, 40))
    ]
    # ["Other", "Mental", "Wood", "Plastic", "Paper"]
    materials = ["Other", "Mental", "Wood", "Plastic", "Paper"]
    data_handler[1]()
    with torch.no_grad():
        for images, targets in data_handler[0]:
            rgb_images = images[:, rgb_order[::-1]].cpu().numpy().transpose(0, 2, 3, 1)
            rgb_images = (rgb_images * 255).astype(np.uint8).copy()
            outputs = model.inference(images)

            for i, (rgb_image, output, target) in enumerate(zip(rgb_images, outputs, targets)):
                raw_rgb = rgb_image.copy()
                if len(output["labels"]) == 0:
                    cv2.putText(
                        rgb_image,
                        "Nothing Here",
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        kind[1],
                        1
                    )
                else:
                    # 绘制预测结果
                    for box, label, score, mask, material, m_score in zip(
                            output["boxes"], output["labels"],
                            output["scores"], output["masks"],
                            output["materials"], output["material_scores"]
                    ):
                        label = label.item()
                        kind = classes[label]
                        mater = materials[material]
                        box = box.cpu().numpy().astype(int)
                        cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), kind[1], 2)
                        cv2.putText(
                            rgb_image,
                            f"{kind[0]} {score:.4f}",
                            (box[0] + 5, box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            kind[1],
                            1
                        )
                        cv2.putText(
                            rgb_image,
                            f"{mater} {m_score:.4f}",
                            (box[0] + 5, box[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            kind[1],
                            1
                        )
                        mask = (mask.cpu().numpy() * 255).astype(np.uint8)[0]
                        overlay = np.zeros_like(rgb_image)
                        overlay[mask > 130] = contrast(kind[1])
                        rgb_image = cv2.addWeighted(rgb_image, 1, overlay, 0.2, 0)

                # 显示图像
                combined = np.concatenate((raw_rgb, rgb_image), axis=1)
                cv2.imshow("output", combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def train(
        model: MatSpecRCNN,
        train_handler: tuple[DataLoader, callable],
        test_handler: tuple[DataLoader, callable],
        filename: str = "mat-spec-rcnn",
        epoches: int = 10,
):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    writer = utils.Writer(f"./logs/{filename}")

    # 训练循环
    model.train()
    count, best = len(train_handler[0]), float("inf")
    for epoch in range(epoches):
        epoch_str = f"{epoch + 1:02d}/{epoches:2d}"
        evaluate(model, test_handler, writer, f"Eva[{epoch_str}]")
        train_handler[1]()

        epoch_loss = 0.0
        process = utils.Processor(total=count, prefix=f"Train[{epoch_str}]")
        for images, targets in train_handler[0]:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()  # noqa
            optimizer.step()

            epoch_loss += losses.item()  # noqa
            process.next(loss=f"{losses:.4f}", total=f"{epoch_loss:.4f}")
            writer.add_scalar("Train Loss.Detail", losses)
        epoch_loss /= count
        process.done(loss=f"{epoch_loss:.4f}")
        writer.add_scalar("Train Loss.Summary", epoch_loss)
        if epoch_loss < best:
            best = epoch_loss
            model.save(f"weights/{filename}.pt")


def main():
    parser = argparse.ArgumentParser("MatSpecRCNN use those parameters:")
    subparsers = parser.add_subparsers(help="running mode")

    # mode: show
    show_parser = subparsers.add_parser(name="show", help="print model summary")
    show_parser.set_defaults(func=main_show)

    # mode: train
    train_parser = subparsers.add_parser(name="train", help="run training")
    train_parser.add_argument(
        "-e", "--epoch", type=int, default=10,
        metavar="\b", help="Number of epoch, default: 10"
    )
    train_parser.add_argument(
        "-b", "--batch", type=int, default=8,
        metavar="\b", help="Batch size, default: 8"
    )
    train_parser.add_argument(
        "-l", "--load", type=str, default=None, metavar="\b",
        help="A weights filename, will be load before train if not None, default: None"
    )
    train_parser.add_argument(
        "-s", "--save", type=str, default="mat-spec-rcnn", metavar="\b",
        help="The weights filename for save action after train, default: mat-spec-rcnn"
    )
    train_parser.set_defaults(func=main_train)

    # mode: infer
    infer_parser = subparsers.add_parser(name="infer", help="run inference")
    infer_parser.add_argument(
        "-l", "--load", type=str, required=True, metavar="\b",
        help="A weights filename, will be load before infer, required"
    )
    infer_parser.add_argument(
        "-b", "--batch", type=int, default=8,
        metavar="\b", help="Batch size, default: 8"
    )
    infer_parser.set_defaults(func=main_infer)

    args = parser.parse_args()
    args.func(args)


def main_show(args: argparse.Namespace):
    print("[Arguments] => mode: show\n")
    model = MatSpecRCNN(7, 5)
    print(model)


def main_train(args: argparse.Namespace):
    print(
        f"[Arguments] => mode: train, epoch: {args.epoch}, "
        f"batch: {args.batch}, load: {args.load}, save: {args.save}\n"
    )
    model = MatSpecRCNN(7, 5)
    if args.load is not None:
        model.load(f"weights/{args.load}.pt")
    model.to(utils.GPU)
    datamaker = Datamaker("images/bg", "images/tar", bg_shape=(384, 512), tar_range=(0.4, 0.8))
    test_handler = datamaker.test(batch=args.batch)
    train_handler = datamaker.train(batch=args.batch)
    train(model, train_handler, test_handler, filename=args.save, epoches=args.epoch)


def main_infer(args: argparse.Namespace):
    print(f"[Arguments] => mode: train, load: {args.load}, batch: {args.batch}\n")
    model = MatSpecRCNN(7, 5)
    model.load(f"weights/{args.load}.pt")
    model.to(utils.GPU)
    datamaker = Datamaker("images/bg", "images/tar", bg_shape=(384, 512), tar_range=(0.4, 0.8))
    test_handler = datamaker.val(batch=args.batch)
    inference(model, test_handler, datamaker.rgb_order)


if __name__ == "__main__":
    main()
