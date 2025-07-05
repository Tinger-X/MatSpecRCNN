import torch
from torch import nn, Tensor
from collections import OrderedDict

from .builder import *
from .transform import Normalizer
from .rio_heads import RoIHeads
from utils import torch_assert

__all__ = ["MatSpecRCNN"]


class MatSpecRCNN(nn.Module):
    """
    Material + Spectral + R-CNN
    """

    def __init__(self, num_classes: int, num_materials: int):
        super(MatSpecRCNN, self).__init__()

        backbone_trainable_layers = 6
        fpn_out_channels = 256
        returned_layers = [1, 2, 3, 4]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        resnet50 = modified_resnet50()

        self.normalizer = Normalizer()
        self.backbone = build_backbone(resnet50, backbone_trainable_layers, return_layers)
        self.fpn = build_fpn(resnet50.inplanes, returned_layers, fpn_out_channels)
        self.rpn = build_rpn(fpn_out_channels)
        self.roi_heads = RoIHeads(num_classes, num_materials, fpn_out_channels)

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        for target in targets:
            boxes = target["boxes"]
            torch_assert(
                isinstance(boxes, torch.Tensor),
                f"Expected target boxes to be of type Tensor, got {type(boxes)}."
            )
            torch_assert(
                boxes.ndim == 2 and boxes.shape[-1] == 4,
                f"Expected target boxes to be a tensor of shape [0] or [N, 4], got {boxes.shape}.",
            )

        images = self.normalizer(images)
        # Check for degenerate boxes
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            if boxes.shape[0] == 0:
                continue
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():  # noqa
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]  # noqa
                degen_bb: list[float] = boxes[bb_idx].tolist()
                torch_assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

        features = self.backbone(images.tensors)
        features = self.fpn(features)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, images: Tensor):
        images = self.normalizer(images)
        features = self.backbone(images.tensors)
        features = self.fpn(features)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.rpn(images, features)
        results = self.roi_heads.inference(features, proposals, images.image_sizes)

        return results

    def save(self, file: str) -> "MatSpecRCNN":
        torch.save(self.state_dict(), file)
        return self

    def load(self, file: str) -> "MatSpecRCNN":
        weights = torch.load(file, weights_only=True)
        self.load_state_dict(weights)
        return self
