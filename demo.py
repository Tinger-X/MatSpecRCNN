import cv2
import numpy as np
from datamaker import RecoMaker, TrackMaker
from datamaker.base import DEVICE, imshow

LABEL_NAMES = ["Unknown", "Tank_48-01", "Tank_81", "Tank_T99", "Air_102", "South_215", "Air_Wright"]
MATERIAL_NAMES = ["Other", "Mental", "Wood", "Plastic", "Paper"]


def test_reco_rgb():
    """目标检测数据集生成器 - RGB三通道示例"""
    reco = RecoMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="rgb",
        bg_shape=(384, 512),
        tar_range=(0.4, 0.8)
    )

    loader, reset_fn = reco.train(batch=4)

    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy()
            rgb_image = image[::-1, :, :].transpose((1, 2, 0))
            rgb_image = (rgb_image * 255).astype("uint8")
            rgb_image = np.ascontiguousarray(rgb_image)

            masks = target["masks"].cpu().detach().numpy()
            boxes = target["boxes"].cpu().detach().numpy().astype(int)
            labels = target["labels"].cpu().detach().numpy()
            materials = target["materials"].cpu().detach().numpy()

            for i in range(len(masks)):
                box = boxes[i]
                cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"{LABEL_NAMES[labels[i]]}|{MATERIAL_NAMES[materials[i]]}"
                cv2.putText(rgb_image, label_text, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            imshow(rgb_image)
        break


def test_reco_540():
    """目标检测数据集生成器 - 540nm单通道示例"""
    reco = RecoMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="540",
        bg_shape=(384, 512),
        tar_range=(0.4, 0.8)
    )

    loader, reset_fn = reco.test(batch=4)

    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy().squeeze()
            image = (image * 255).astype("uint8")
            image = np.ascontiguousarray(image)

            masks = target["masks"].cpu().detach().numpy()
            boxes = target["boxes"].cpu().detach().numpy().astype(int)

            for i in range(len(masks)):
                box = boxes[i]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 255, 2)

            imshow(image)
        break


def test_reco_full():
    """目标检测数据集生成器 - 全九通道示例"""
    reco = RecoMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="full",
        bg_shape=(384, 512),
        tar_range=(0.4, 0.8)
    )

    loader, reset_fn = reco.val(batch=4)

    for images, targets in loader:
        for image, target in zip(images, targets):
            image = image.cpu().detach().numpy()
            rgb_image = image[reco.rgb_order[::-1]].transpose((1, 2, 0))
            rgb_image = (rgb_image * 255).astype("uint8")
            rgb_image = np.ascontiguousarray(rgb_image)

            masks = target["masks"].cpu().detach().numpy()
            boxes = target["boxes"].cpu().detach().numpy().astype(int)
            labels = target["labels"].cpu().detach().numpy()

            for i in range(len(masks)):
                box = boxes[i]
                cv2.rectangle(rgb_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(rgb_image, LABEL_NAMES[labels[i]], (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            imshow(rgb_image)
        break


def test_reco_custom():
    """目标检测数据集生成器 - 自定义数据集大小和随机种子"""
    reco = RecoMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="rgb",
        bg_shape=(480, 640),
        tar_range=(0.3, 0.6)
    )

    loader, reset_fn = reco.any(batch=2, size=100, seed=42)

    print(f"Device: {DEVICE}")
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Batches: {len(loader)}")

    for batch_idx, (images, targets) in enumerate(loader):
        print(f"Batch {batch_idx}: images shape {images.shape}")
        if batch_idx >= 2:
            break


def test_track_sequence():
    """目标跟踪数据集生成器 - 序列可视化示例"""
    trackmaker = TrackMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="full",
        seq_len=60,
        bg_shape=(384, 512),
        tar_range=(0.1, 0.25)
    )

    loader, reset_fn = trackmaker.train(batch=2)

    for images, targets in loader:
        for seq_idx in range(len(images)):
            seq_image = images[seq_idx].cpu().detach().numpy()
            seq_rgb = seq_image[:, trackmaker.rgb_order[::-1]].transpose((0, 2, 3, 1))
            seq_rgb = (seq_rgb * 255).astype("uint8")
            seq_rgb = np.ascontiguousarray(seq_rgb)

            boxes = targets[seq_idx]["boxes"].cpu().detach().numpy().astype(int)
            labels = targets[seq_idx]["labels"].cpu().detach().numpy()

            for t in range(len(seq_rgb)):
                frame = seq_rgb[t].copy()
                box = boxes[t]

                if len(box) == 4:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{LABEL_NAMES[labels[t]]} Frame:{t + 1}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("Tracking", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        break

    cv2.destroyAllWindows()


def test_track_statistics():
    """目标跟踪数据集生成器 - 数据集统计信息"""
    trackmaker = TrackMaker(
        bg_path="./data/bg",
        tar_path="./data/tar",
        data_type="rgb",
        seq_len=50,
        bg_shape=(384, 512),
        tar_range=(0.1, 0.3)
    )

    train_loader, _ = trackmaker.train(batch=4)
    test_loader, _ = trackmaker.test(batch=4)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Sequence length: {trackmaker.seq_len}")
    print(f"Channel index: {trackmaker._idx}")

    for images, targets in train_loader:
        print(f"\nBatch images shape: {images.shape}")
        print(f"Batch masks shape: {targets[0]['masks'].shape}")
        print(f"Batch boxes shape: {targets[0]['boxes'].shape}")
        print(f"Batch labels: {targets[0]['labels']}")
        break


if __name__ == "__main__":
    # print("=" * 60)
    # print("目标检测数据集示例 (RGB)")
    # print("=" * 60)
    # test_reco_rgb()
    #
    # print("\n" + "=" * 60)
    # print("目标检测数据集示例 (540nm)")
    # print("=" * 60)
    # test_reco_540()
    #
    # print("\n" + "=" * 60)
    # print("目标检测数据集示例 (全通道)")
    # print("=" * 60)
    # test_reco_full()
    #
    # print("\n" + "=" * 60)
    # print("目标检测数据集示例 (自定义配置)")
    # print("=" * 60)
    # test_reco_custom()

    print("\n" + "=" * 60)
    print("目标跟踪数据集示例 (序列可视化)")
    print("=" * 60)
    test_track_sequence()

    # print("\n" + "=" * 60)
    # print("目标跟踪数据集示例 (数据集统计)")
    # print("=" * 60)
    # test_track_statistics()
