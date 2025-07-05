import torch
from torch import nn
from model.network import MatSpecRCNN
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model.transform import Normalizer

__all__ = [
    "MatSpecRCNN_Full", "MatSpecRCNN_RGB", "MatSpecRCNN_540", "MaskRCNN_ResNet50"
]

MatSpecRCNN_Full = MatSpecRCNN


class MatSpecRCNN_RGB(MatSpecRCNN):
    def __init__(self, num_classes: int, num_materials: int):
        super(MatSpecRCNN_RGB, self).__init__(num_classes, num_materials)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.normalizer = Normalizer(
            mean=[0.2339738, 0.061436515, 0.23215543],
            std=[0.14639099, 0.056200337, 0.1462142]
        )


class MatSpecRCNN_540(MatSpecRCNN):
    def __init__(self, num_classes: int, num_materials: int):
        super(MatSpecRCNN_540, self).__init__(num_classes, num_materials)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.normalizer = Normalizer(
            mean=[0.21487042],
            std=[0.15836369]
        )


class MaskRCNN_ResNet50(MaskRCNN):
    def __init__(self, num_classes: int):
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            norm_layer=nn.BatchNorm2d,
            trainable_layers=5
        )
        super(MaskRCNN_ResNet50, self).__init__(backbone=backbone, num_classes=num_classes)

    def inference(self, images: torch.Tensor) -> torch.Tensor:
        return super(MaskRCNN_ResNet50, self).forward(images)  # noqa

    def save(self, file: str) -> "MaskRCNN_ResNet50":
        torch.save(self.state_dict(), file)
        return self

    def load(self, file: str) -> "MaskRCNN_ResNet50":
        weights = torch.load(file, weights_only=True)
        self.load_state_dict(weights)
        return self


def show():
    """
    usage:  "MatSpecRCNN_Full", "MatSpecRCNN_RGB", "MatSpecRCNN_540", "MaskRCNN_ResNet50"
        # MaskRCNN_ResNet50
        python models.py MaskRCNN_ResNet50 > ../logs/mask-rcnn/model.txt
        # MatSpecRCNN_Full
        python models.py MatSpecRCNN_Full > ../logs/self-full/model.txt
        # MatSpecRCNN_RGB
        python models.py MatSpecRCNN_RGB > ../logs/self-rgb/model.txt
        # MatSpecRCNN_540
        python models.py MatSpecRCNN_540 > ../logs/self-540/model.txt
    """
    import sys

    if len(sys.argv) != 2:
        return print(f"usage: python models.py <name: {' | '.join(__all__)}>")
    if sys.argv[1] not in __all__:
        return print(f"got an unkown model name, expect: {__all__}")
    model_cls = eval(sys.argv[1])
    model_args = (7, 5)
    if sys.argv[1] == "MaskRCNN_ResNet50":
        model_args = (7,)
    model = model_cls(*model_args)
    print(model)


if __name__ == "__main__":
    show()
