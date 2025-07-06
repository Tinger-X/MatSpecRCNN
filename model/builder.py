from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck  # noqa
from torchvision.models._utils import IntermediateLayerGetter  # noqa
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork

__all__ = ["modified_resnet50", "build_backbone", "build_fpn", "build_rpn"]


def modified_resnet50() -> ResNet:
    resnet = ResNet(Bottleneck, [3, 4, 6, 3])
    resnet.conv1 = nn.Conv2d(
        9, 64, kernel_size=7,
        stride=2, padding=3, bias=False
    )

    return resnet


def build_backbone(model: ResNet, trainable_layers: int, return_layers: dict) -> IntermediateLayerGetter:
    if trainable_layers < 0 or trainable_layers > 6:
        raise ValueError(f"Trainable layers should be in the range [0, 6], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1", "bn1"][:trainable_layers]
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return IntermediateLayerGetter(model, return_layers=return_layers)


def build_fpn(inplanes: int, returned_layers: list[int], out_channels: int) -> FeaturePyramidNetwork:
    in_channels_stage2 = inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]

    return FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


def build_rpn(fpn_out_channels: int) -> RegionProposalNetwork:
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RPNHead(fpn_out_channels, anchor_generator.num_anchors_per_location()[0])

    nms_thresh = 0.7
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 256
    positive_fraction = 0.5
    score_thresh = 0.0
    pre_nms_top_n_train = 2000
    pre_nms_top_n_test = 1000
    post_nms_top_n_train = 2000
    post_nms_top_n_test = 1000
    pre_nms_top_n = dict(training=pre_nms_top_n_train, testing=pre_nms_top_n_test)
    post_nms_top_n = dict(training=post_nms_top_n_train, testing=post_nms_top_n_test)

    return RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=head,
        fg_iou_thresh=fg_iou_thresh,
        bg_iou_thresh=bg_iou_thresh,
        batch_size_per_image=batch_size_per_image,
        positive_fraction=positive_fraction,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh
    )
