"""
usage:
```bash
    # run in NVIDIA GeForce RTX 3060
    export PYTHONPATH=/path/to/MatSpecRCNN
    cd /path/to/MatSpecRCNN/metrics
    # MaskRCNN
    python -u infer.py -m mask-rcnn -b 8
    # MatSpecRCNN-RGB
    python -u infer.py -m self-rgb -b 8
    # MatSpecRCNN-540
    python -u infer.py -m self-540 -b 8
    # MatSpecRCNN-Full
    # see in /path/to/MatSpecRCNN/main.py
```
"""


import cv2
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

import utils
import models
from datamaker import Datamaker


def infer(model, dataloader: DataLoader, image_handler: callable, has_material: bool):
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
    with torch.no_grad():
        for images, targets in dataloader:
            rgb_images = image_handler(images)
            outputs = model.inference(images)

            for rgb_image, output, target in zip(rgb_images, outputs, targets):
                raw_rgb = rgb_image.copy()
                count = len(output["labels"])
                if count == 0:
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
                    for j in range(count):
                        box = output["boxes"][j]
                        label = output["labels"][j].item()
                        score = output["scores"][j]
                        mask = output["masks"][j]

                        kind = classes[label]
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
                        if has_material:
                            mater = materials[output["materials"][j]]
                            m_score = output["material_scores"][j]
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


def args_parse():
    parser = argparse.ArgumentParser("Metrics Model Inference")
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        choices=["mask-rcnn", "self-rgb", "self-540"], help="Model name"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=8, help="Training batch size"
    )
    args = parser.parse_args()
    print(
        f"[Arguments] => model: {args.model}, batch: {args.batch}\n"
    )
    return args


def image_handler_rgb(images: torch.Tensor) -> np.ndarray:
    return (images[:, [2, 1, 0], :, :] * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def image_handler_540(images: torch.Tensor) -> np.ndarray:
    return (images.squeeze(1) * 255).to(torch.uint8).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().cpu().numpy()


def main():
    args = args_parse()
    model_args = (7, 5)
    data_type = "rgb"
    image_handler = image_handler_rgb
    material = True
    if args.model == "mask-rcnn":
        model_args = (7,)
        model_cls = models.MaskRCNN_ResNet50
        material = False
    elif args.model == "self-rgb":
        model_cls = models.MatSpecRCNN_RGB
    else:
        model_cls = models.MatSpecRCNN_540
        data_type = "540"
        image_handler = image_handler_540
    datamaker = Datamaker(
        "../images/bg", "../images/tar", data_type=data_type,  # noqa
        bg_shape=(384, 512), tar_range=(0.4, 0.8)
    )
    model = model_cls(*model_args)
    model.load(f"../weights/{args.model}.pt")
    model.to(utils.GPU)
    val_loader, _ = datamaker.val(batch=args.batch)
    infer(model, val_loader, image_handler, material)


if __name__ == "__main__":
    main()
