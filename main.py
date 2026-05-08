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
from datamaker import RecoMaker, TrackMaker


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


def _draw_detection(rgb_image, output, classes, materials, contrast_func):
    """绘制检测结果的通用函数"""
    if len(output["labels"]) == 0:
        cv2.putText(
            rgb_image,
            "No Detection",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        return rgb_image

    for box, label, score, mask, material, m_score in zip(
            output["boxes"], output["labels"],
            output["scores"], output["masks"],
            output["materials"], output["material_scores"]
    ):
        label_val = label.item()
        material_val = material.item()
        kind = classes[label_val]
        mater = materials[material_val]
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
        # cv2.putText(
        #     rgb_image,
        #     f"{mater} {m_score:.4f}",
        #     (box[0] + 5, box[1] + 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     kind[1],
        #     1
        # )

        mask = (mask.cpu().numpy() * 255).astype(np.uint8)[0]
        overlay = np.zeros_like(rgb_image)
        overlay[mask > 130] = contrast_func(kind[1])
        rgb_image = cv2.addWeighted(rgb_image, 1, overlay, 0.2, 0)

    return rgb_image


def inference(
        model: MatSpecRCNN,
        data_handler: tuple[DataLoader, callable],
        rgb_order: list[int],
        is_sequence: bool = False
):
    """
    统一的推理函数，支持单帧图像和视频序列

    Args:
        model: 模型实例
        data_handler: 数据加载器和重置函数
        rgb_order: RGB通道顺序
        is_sequence: 是否为视频序列模式
    """
    import time

    def contrast(color):
        return 255 - color[0], 255 - color[1], 255 - color[2]

    model.eval()
    classes = [
        ("Unknown", (0, 0, 255)), ("Tank_48-01", (20, 20, 255)),
        ("Tank_81", (30, 30, 255)), ("Tank_T99", (40, 40, 255)),
        ("Air_102", (20, 255, 20)), ("South_215", (30, 255, 30)),
        ("Air_Wright", (40, 255, 40))
    ]
    materials = ["Other", "Mental", "Wood", "Plastic", "Paper"]

    data_loader, reset_func = data_handler
    reset_func()

    if is_sequence:
        # 视频序列模式的统计变量
        total_frames = 0
        total_time = 0.0
        total_detections = 0
        correct_labels = 0
        correct_materials = 0

    with torch.no_grad():
        for images, targets in data_loader:
            if is_sequence:
                # 序列模式: images [batch, seq_len, C, H, W]
                batch_size = images.shape[0]
                seq_len = images.shape[1]

                for batch_idx in range(batch_size):
                    sequence = images[batch_idx]  # [seq_len, C, H, W]
                    target = targets[batch_idx]

                    rgb_sequence = sequence[:, rgb_order[::-1]].cpu().numpy().transpose(0, 2, 3, 1)
                    rgb_sequence = (rgb_sequence * 255).astype(np.uint8)

                    # 批量推理：将序列分成多个batch进行推理（最大16帧一个batch）
                    max_infer_batch = 8
                    all_outputs = []
                    batch_inference_times = []

                    for start_idx in range(0, seq_len, max_infer_batch):
                        end_idx = min(start_idx + max_infer_batch, seq_len)
                        frame_batch = sequence[start_idx:end_idx]  # [batch_size, C, H, W]

                        start_time = time.time()
                        batch_outputs = model.inference(frame_batch)
                        batch_time = time.time() - start_time

                        batch_inference_times.append(batch_time)
                        all_outputs.extend(batch_outputs)

                        total_frames += (end_idx - start_idx)
                        total_time += batch_time

                    # 逐帧显示结果
                    for frame_idx in range(seq_len):
                        raw_rgb = rgb_sequence[frame_idx].copy()
                        result_rgb = raw_rgb.copy()

                        output = all_outputs[frame_idx]
                        gt_label = target["labels"][frame_idx].item()
                        gt_material = target["materials"][frame_idx].item()
                        gt_box = target["boxes"][frame_idx].cpu().numpy().astype(int)

                        # 绘制ground truth (绿色)
                        if len(gt_box) == 4:
                            cv2.rectangle(raw_rgb, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
                            cv2.putText(
                                raw_rgb,
                                f"GT: {classes[gt_label][0]}",
                                (gt_box[0] + 5, gt_box[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1
                            )
                            # cv2.putText(
                            #     raw_rgb,
                            #     f"GT: {materials[gt_material]}",
                            #     (gt_box[0] + 5, gt_box[1] + 40),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.5,
                            #     (0, 255, 0),
                            #     1
                            # )

                        # 绘制预测结果并统计准确率
                        if len(output["labels"]) > 0:
                            total_detections += len(output["labels"])
                            for label, material in zip(output["labels"], output["materials"]):
                                if label.item() == gt_label:
                                    correct_labels += 1
                                if material.item() == gt_material:
                                    correct_materials += 1

                        result_rgb = _draw_detection(result_rgb, output, classes, materials, contrast)

                        # 添加性能信息（使用平均FPS，因为是批量推理）
                        avg_fps = total_frames / total_time if total_time > 0 else 0
                        label_acc = correct_labels / total_detections * 100 if total_detections > 0 else 0
                        material_acc = correct_materials / total_detections * 100 if total_detections > 0 else 0

                        info_text = [
                            f"Frame: {frame_idx + 1}/{seq_len}",
                            f"Avg FPS: {avg_fps:.2f} (Batch: {max_infer_batch})",
                            f"Label Acc: {label_acc:.2f}%",
                            # f"Material Acc: {material_acc:.2f}%"
                        ]

                        for i, text in enumerate(info_text):
                            cv2.putText(
                                result_rgb,
                                text,
                                (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2
                            )

                        combined = np.concatenate((raw_rgb, result_rgb), axis=1)
                        cv2.imshow("Tracking: Ground Truth (Left) | Prediction (Right)", combined)

                        key = cv2.waitKey(30)
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            print(f"\n[Summary] Total Frames: {total_frames}, Avg FPS: {avg_fps:.2f}")
                            print(f"[Summary] Label Accuracy: {label_acc:.2f}%, Material Accuracy: {material_acc:.2f}%")
                            return
                        elif key == ord('p'):
                            cv2.waitKey(0)

                    cv2.destroyAllWindows()

                # 最终统计
                avg_fps = total_frames / total_time if total_time > 0 else 0
                label_acc = correct_labels / total_detections * 100 if total_detections > 0 else 0
                material_acc = correct_materials / total_detections * 100 if total_detections > 0 else 0

                print(f"\n[Final Summary]")
                print(f"Total Frames: {total_frames}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Total Detections: {total_detections}")
                print(f"Label Accuracy: {label_acc:.2f}%")
                print(f"Material Accuracy: {material_acc:.2f}%")

            else:
                # 单帧模式: images [batch, C, H, W]
                rgb_images = images[:, rgb_order[::-1]].cpu().numpy().transpose(0, 2, 3, 1)
                rgb_images = (rgb_images * 255).astype(np.uint8).copy()

                start_time = time.time()
                outputs = model.inference(images)
                inference_time = time.time() - start_time

                batch_size = len(outputs)
                fps = batch_size / inference_time if inference_time > 0 else 0

                for idx, (rgb_image, output, target) in enumerate(zip(rgb_images, outputs, targets)):
                    raw_rgb = rgb_image.copy()

                    # 获取ground truth (可能有多个对象)
                    gt_labels = target["labels"].cpu().numpy()
                    gt_materials = target["materials"].cpu().numpy()
                    gt_boxes = target["boxes"].cpu().numpy().astype(int)

                    # 统计准确率 (简化：比较第一个预测和第一个GT)
                    num_gt = len(gt_labels)
                    num_pred = len(output["labels"])
                    label_correct = False
                    material_correct = False

                    if num_gt > 0 and num_pred > 0:
                        pred_label = output["labels"][0].item()
                        pred_material = output["materials"][0].item()
                        label_correct = (pred_label == gt_labels[0])
                        material_correct = (pred_material == gt_materials[0])

                    result_rgb = _draw_detection(rgb_image.copy(), output, classes, materials, contrast)

                    # 打印性能指标（覆盖打印）
                    print(
                        f"\rImage: {idx + 1}/{batch_size} | "
                        f"FPS: {fps:.2f} (Batch: {batch_size}) | "
                        f"Inference: {inference_time * 1000:.2f}ms",
                        end=""
                    )

                    combined = np.concatenate((raw_rgb, result_rgb), axis=1)
                    cv2.imshow("Inference: Ground Truth (Left) | Prediction (Right)", combined)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                print()  # 换行


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
    show_parser.set_defaults(func=endpoint_show)

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
    train_parser.set_defaults(func=endpoint_train)

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
    infer_parser.set_defaults(func=endpoint_infer)

    # mode: track
    track_parser = subparsers.add_parser(name="track", help="run tracking")
    track_parser.add_argument(
        "-l", "--load", type=str, required=True, metavar="\b",
        help="A weights filename, will be load before track, required"
    )
    track_parser.add_argument(
        "-b", "--batch", type=int, default=8,
        metavar="\b", help="Batch size, default: 8"
    )
    track_parser.add_argument(
        "-s", "--len", type=int, default=60,
        metavar="\b", help="Sequence length, default: 60"
    )
    track_parser.set_defaults(func=endpoint_track)

    args = parser.parse_args()
    args.func(args)


def endpoint_show(args: argparse.Namespace):
    print("[Arguments] => mode: show\n")
    model = MatSpecRCNN(7, 5)
    print(model)


def endpoint_train(args: argparse.Namespace):
    print(
        f"[Arguments] => mode: train, epoch: {args.epoch}, "
        f"batch: {args.batch}, load: {args.load}, save: {args.save}\n"
    )
    model = MatSpecRCNN(7, 5)
    if args.load is not None:
        model.load(f"weights/{args.load}.pt")
    model.to(utils.GPU)
    datamaker = RecoMaker("data/bg", "data/tar", bg_shape=(384, 512), tar_range=(0.4, 0.8))
    test_handler = datamaker.test(batch=args.batch)
    train_handler = datamaker.train(batch=args.batch)
    train(model, train_handler, test_handler, filename=args.save, epoches=args.epoch)


def endpoint_infer(args: argparse.Namespace):
    print(f"[Arguments] => mode: infer, load: {args.load}, batch: {args.batch}\n")
    model = MatSpecRCNN(7, 5)
    model.load(f"weights/{args.load}.pt")
    model.to(utils.GPU)
    datamaker = RecoMaker("data/bg", "data/tar", bg_shape=(384, 512), tar_range=(0.4, 0.8))
    test_handler = datamaker.val(batch=args.batch)
    inference(model, test_handler, datamaker.rgb_order, is_sequence=False)


def endpoint_track(args: argparse.Namespace):
    print(f"[Arguments] => mode: track, load: {args.load}, batch: {args.batch}, seq-len: {args.len}\n")
    model = MatSpecRCNN(7, 5)
    model.load(f"weights/{args.load}.pt")
    model.to(utils.GPU)
    datamaker = TrackMaker(
        "data/bg", "data/tar",
        seq_len=args.len, bg_shape=(384, 512), tar_range=(0.4, 0.8)
    )
    test_handler = datamaker.val(batch=args.batch)
    inference(model, test_handler, datamaker.rgb_order, is_sequence=True)


if __name__ == "__main__":
    main()
