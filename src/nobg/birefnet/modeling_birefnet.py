import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import SwinConfig as _SwinConfig, SwinModel

from ..mixin import Revised_Mixin
from ..utils import model_card_template


@dataclass
class BiRefNetConfig:
    image_size: int = 1024
    patch_size: int = 4
    embed_dim: int = 96
    depths: list = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: list = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 8
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    dec_channels_inter: int = 64
    use_multi_scale_input: bool = True
    use_gradient_attention: bool = True
    use_image_patch_injection: bool = True


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 256):
        super().__init__()
        dilations = [1, 6, 12, 18]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=dilations[3],
                dilation=dilations[3],
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(inter_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=True)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.aspp = ASPP(inter_channels, inter_channels, inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.aspp(x)
        x = self.bn_out(self.conv_out(x))
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, channels: list[int], config: BiRefNetConfig):
        super().__init__()
        inter = config.dec_channels_inter

        # channels = [C1, C2, C3, C4] from shallowest to deepest
        # decoder goes top-down: x4 -> x3 -> x2 -> x1
        # dec_out_channels[i] = channels of decoder output at stage i
        # stage 0: deepest->next, stage 1: next->next, stage 2: shallowest
        dec_out = [channels[2], channels[1], channels[0]]

        self.squeeze = BasicDecBlk(channels[3], channels[3], inter)

        self.upsample_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[3], dec_out[0], 1),
                nn.Conv2d(dec_out[0], dec_out[1], 1),
                nn.Conv2d(dec_out[1], dec_out[2], 1),
            ]
        )

        self.lateral_blks = nn.ModuleList(
            [
                BasicLatBlk(channels[2], dec_out[0]),
                BasicLatBlk(channels[1], dec_out[1]),
                BasicLatBlk(channels[0], dec_out[2]),
            ]
        )

        self.decoder_blks = nn.ModuleList(
            [
                BasicDecBlk(dec_out[0], dec_out[0], inter),
                BasicDecBlk(dec_out[1], dec_out[1], inter),
                BasicDecBlk(dec_out[2], dec_out[2], inter),
            ]
        )

        self.side_convs = nn.ModuleList(
            [
                nn.Conv2d(dec_out[0], 1, 1),
                nn.Conv2d(dec_out[1], 1, 1),
                nn.Conv2d(dec_out[2], 1, 1),
            ]
        )

        if config.use_image_patch_injection:
            self.patch_convs = nn.ModuleList(
                [
                    SimpleConvs(3, dec_out[0]),
                    SimpleConvs(3, dec_out[1]),
                    SimpleConvs(3, dec_out[2]),
                ]
            )

        if config.use_gradient_attention:
            self.gdt_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(dec_out[i], 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                    )
                    for i in range(3)
                ]
            )
            self.gdt_attns = nn.ModuleList([nn.Conv2d(16, 1, 1) for _ in range(3)])

        self.final_conv = nn.Conv2d(dec_out[2], 1, 1)
        self.use_gradient_attention = config.use_gradient_attention
        self.use_image_patch_injection = config.use_image_patch_injection

    def forward(
        self, x: torch.Tensor, features: list[torch.Tensor]
    ) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
        x1, x2, x3, x4 = features
        laterals = [x3, x2, x1]

        p = self.squeeze(x4)
        intermediate_logits: list[torch.Tensor] = []

        for i in range(3):
            p_up = F.interpolate(
                p, size=laterals[i].shape[2:], mode="bilinear", align_corners=True
            )
            p_up = self.upsample_convs[i](p_up)
            lat = self.lateral_blks[i](laterals[i])
            combined = p_up + lat

            if self.use_image_patch_injection:
                img_down = F.interpolate(
                    x, size=combined.shape[2:], mode="bilinear", align_corners=True
                )
                combined = combined + self.patch_convs[i](img_down)

            p = self.decoder_blks[i](combined)

            side = self.side_convs[i](p)
            intermediate_logits.append(side)

            if self.use_gradient_attention:
                gdt_feat = self.gdt_convs[i](p)
                attn = self.gdt_attns[i](gdt_feat).sigmoid()
                p = p * attn

        p = F.interpolate(p, size=x.shape[2:], mode="bilinear", align_corners=True)
        logits = self.final_conv(p)

        intermediate_logits = [
            F.interpolate(s, size=x.shape[2:], mode="bilinear", align_corners=True)
            for s in intermediate_logits
        ]

        return {"logits": logits, "intermediate_logits": intermediate_logits}


class BiRefNet(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    tags=["nobg", "birefnet"],
    model_card_template=model_card_template(
        class_name="BiRefNet", default_repo="nobg/birefnet"
    ),
):
    """Bilateral Reference Network for image segmentation and background removal."""

    def __init__(self, config: Optional[BiRefNetConfig] = None):
        super().__init__()
        self.config = config or BiRefNetConfig()

        swin_config = _SwinConfig(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            depths=self.config.depths,
            num_heads=self.config.num_heads,
            window_size=self.config.window_size,
            mlp_ratio=self.config.mlp_ratio,
            drop_path_rate=self.config.drop_path_rate,
        )
        self.backbone = SwinModel(swin_config)

        num_stages = len(self.config.depths)
        base_channels = [self.config.embed_dim * (2**i) for i in range(num_stages)]

        if self.config.use_multi_scale_input:
            channels = [c * 2 for c in base_channels]
        else:
            channels = base_channels

        context_in = sum(channels)
        self.context_conv = nn.Conv2d(context_in, channels[-1], 1)

        self.decoder = Decoder(channels, self.config)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
        out = self.backbone(pixel_values, output_hidden_states=True)
        features_full = list(out.reshaped_hidden_states[:4])

        if self.config.use_multi_scale_input:
            half = F.interpolate(
                pixel_values, scale_factor=0.5, mode="bilinear", align_corners=True
            )
            out_half = self.backbone(half, output_hidden_states=True)
            features_half = list(out_half.reshaped_hidden_states[:4])
            features = [
                torch.cat(
                    [
                        full,
                        F.interpolate(
                            half_f,
                            size=full.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                    ],
                    dim=1,
                )
                for full, half_f in zip(features_full, features_half)
            ]
        else:
            features = features_full

        x1, x2, x3, x4 = features

        x1_down = F.interpolate(
            x1, size=x4.shape[2:], mode="bilinear", align_corners=True
        )
        x2_down = F.interpolate(
            x2, size=x4.shape[2:], mode="bilinear", align_corners=True
        )
        x3_down = F.interpolate(
            x3, size=x4.shape[2:], mode="bilinear", align_corners=True
        )
        context = torch.cat([x1_down, x2_down, x3_down, x4], dim=1)
        x4 = self.context_conv(context)

        return self.decoder(pixel_values, [x1, x2, x3, x4])
