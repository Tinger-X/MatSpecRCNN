import torch
from torch import nn
import torch.nn.functional as F  # noqa
from torchvision.ops import roi_align, MultiScaleRoIAlign

__all__ = ["MaterialRoIPool", "MaterialHead", "MaterialPredictor"]


class MaterialRoIPool(nn.Module):
    """
    Combines local RoIAlign with global context pooling for richer material features.
    """

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MaterialRoIPool, self).__init__()
        self.featmap_names = featmap_names
        self.local_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )

    def forward(self, features, proposals, image_shapes):
        return self._with_global_context(features, proposals, image_shapes)

    def _remove_global_context(self, features, proposals, image_shapes):
        return self.local_pool(features, proposals, image_shapes)

    def _with_global_context(self, features, proposals, image_shapes):
        local_feats = self.local_pool(features, proposals, image_shapes)  # [N,C,P,P]
        # global context from highest-resolution (first) fmap
        context = []
        for i, bbox in enumerate(proposals):
            fmap = features[self.featmap_names[0]][i:i + 1]
            ctx = roi_align(fmap, [bbox], output_size=(1, 1), spatial_scale=1.0)
            context.append(ctx)
        context = torch.cat(context, dim=0)  # [N,C,1,1]
        context = context.expand(-1, -1, local_feats.size(-2), local_feats.size(-1))
        return local_feats + context


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.spat = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out * self.spat(out)


class SpectralAttention(nn.Module):
    """
    Applies frequency-domain attention by FFT magnitude pooling and channel gating.
    """

    def __init__(self, channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.spec = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [N,C,H,W]
        freq = torch.fft.rfft2(x, dim=(-2, -1))
        mag = torch.abs(freq)
        desc = mag.mean(dim=(-2, -1))  # [N,C]
        weights = self.spec(desc).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class MaterialHead(nn.Module):
    """
    Double-S Attention Material Head, with 1×1 conv reduction + global pooling
    """

    def __init__(self, in_channels, representation_size, hidden_size=None):
        super(MaterialHead, self).__init__()
        hidden_size = hidden_size or representation_size // 2

        # spatial & spectral attention
        self.spatial_attn = SpatialAttention(in_channels)
        self.spectral_attn = SpectralAttention(in_channels)

        # 1×1 conv for channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # global pooling → feature vector
        )

        # projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, representation_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(representation_size, representation_size, bias=True),
        )

    def forward(self, x):
        return self._with_attn(x)

    def _remove_spatial_attn(self, x):
        x = self.spectral_attn(x)
        x = self.reduce(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

    def _remove_spectral_attn(self, x):
        x = self.spatial_attn(x)
        x = self.reduce(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

    def _remove_attn(self, x):
        x = self.reduce(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

    def _with_attn(self, x):
        x = self.spatial_attn(x)
        x = self.spectral_attn(x)
        x = self.reduce(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


class MaterialPredictor(nn.Module):
    """
    Material classification predictor head. Takes pooled features and outputs logits over materials.
    """

    def __init__(self, in_channels, num_materials):
        super(MaterialPredictor, self).__init__()

        self.layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_materials)
        )

    def forward(self, features):
        # features: Tensor of shape (N, C, pool_size, pool_size)
        x = features.flatten(start_dim=1)
        return self.layers(x)
