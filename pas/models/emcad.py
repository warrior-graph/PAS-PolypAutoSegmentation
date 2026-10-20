"""EMCAD decoder (Rahman & Marculescu, CVPR 2024) on top of a timm PVTv2 encoder.

Reference: "EMCAD: Efficient Multi-scale Convolutional Attention Decoding for
Medical Image Segmentation", arXiv:2405.06880.
"""

from math import gcd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "relu6": nn.ReLU6, "gelu": nn.GELU}[name](
        **({} if name == "gelu" else {"inplace": True})
    )


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class MSDC(nn.Module):
    """Multi-scale depth-wise convolutions applied in parallel."""

    def __init__(self, channels, kernel_sizes, activation="relu6"):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                _act(activation),
            )
            for k in kernel_sizes
        )

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)


class MSCB(nn.Module):
    """Multi-scale convolution block: expand -> multi-scale DW conv -> shuffle -> project."""

    def __init__(self, in_ch, out_ch, kernel_sizes=(1, 3, 5), expansion=2, activation="relu6"):
        super().__init__()
        ex_ch = int(in_ch * expansion)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_ch, ex_ch, 1, bias=False), nn.BatchNorm2d(ex_ch), _act(activation)
        )
        self.msdc = MSDC(ex_ch, kernel_sizes, activation)
        self.pconv2 = nn.Sequential(nn.Conv2d(ex_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.shuffle_groups = gcd(ex_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.msdc(self.pconv1(x))
        out = self.pconv2(channel_shuffle(out, self.shuffle_groups))
        return self.skip(x) + out


class EUCB(nn.Module):
    """Efficient up-convolution block: x2 upsample -> DW conv -> PW conv."""

    def __init__(self, in_ch, out_ch, activation="relu6"):
        super().__init__()
        self.dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            _act(activation),
        )
        self.pwc = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pwc(self.dwc(x))


class LGAG(nn.Module):
    """Large-kernel grouped attention gate fusing a skip connection with the decoder signal."""

    def __init__(self, channels, kernel_size=3, activation="relu"):
        super().__init__()
        inter = max(channels // 2, 1)
        groups = inter

        def gconv():
            return nn.Sequential(
                nn.Conv2d(channels, inter, kernel_size, padding=kernel_size // 2, groups=groups, bias=False),
                nn.BatchNorm2d(inter),
            )

        self.w_g, self.w_x = gconv(), gconv()
        self.act = _act(activation)
        self.psi = nn.Sequential(nn.Conv2d(inter, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(self.act(self.w_g(g) + self.w_x(x)))


class CAB(nn.Module):
    """Channel attention block; returns attention weights."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x):
        avg = self.fc(F.adaptive_avg_pool2d(x, 1))
        mx = self.fc(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg + mx)


class SAB(nn.Module):
    """Spatial attention block; returns attention weights."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


class EMCADDecoder(nn.Module):
    def __init__(self, channels, kernel_sizes=(1, 3, 5), expansion=2, lgag_kernel=3, activation="relu6"):
        """`channels` are ordered deepest first, e.g. [512, 320, 128, 64]."""
        super().__init__()
        c = list(channels)
        self.cabs = nn.ModuleList(CAB(ch) for ch in c)
        self.sab = SAB()
        self.mscbs = nn.ModuleList(MSCB(ch, ch, kernel_sizes, expansion, activation) for ch in c)
        self.eucbs = nn.ModuleList(EUCB(c[i], c[i + 1], activation) for i in range(len(c) - 1))
        self.lgags = nn.ModuleList(LGAG(c[i + 1], lgag_kernel) for i in range(len(c) - 1))

    def _refine(self, i, d):
        d = self.cabs[i](d) * d
        d = self.sab(d) * d
        return self.mscbs[i](d)

    def forward(self, x, skips):
        """x: deepest feature; skips: remaining encoder features, deepest first."""
        d = self._refine(0, x)
        outs = [d]
        for i, skip in enumerate(skips):
            d = self.eucbs[i](d)
            d = d + self.lgags[i](g=d, x=skip)
            d = self._refine(i + 1, d)
            outs.append(d)
        return outs


class EMCADNet(nn.Module):
    """PVTv2 encoder + EMCAD decoder with one prediction head per decoder stage.

    forward() returns a list of logits at input resolution, deepest (coarsest) first;
    the last element is the main prediction.
    """

    def __init__(
        self,
        encoder="pvt_v2_b2",
        pretrained=True,
        num_classes=1,
        kernel_sizes=(1, 3, 5),
        expansion=2,
        lgag_kernel=3,
        activation="relu6",
    ):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        enc_ch = self.encoder.feature_info.channels()[::-1]
        self.decoder = EMCADDecoder(enc_ch, kernel_sizes, expansion, lgag_kernel, activation)
        self.heads = nn.ModuleList(nn.Conv2d(ch, num_classes, 1) for ch in enc_ch)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)[::-1]
        dec = self.decoder(feats[0], feats[1:])
        return [
            F.interpolate(head(d), size=size, mode="bilinear", align_corners=False)
            for head, d in zip(self.heads, dec)
        ]
