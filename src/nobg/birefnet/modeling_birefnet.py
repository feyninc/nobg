import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union

from torchvision.ops import deform_conv2d

from ..loss import birefnet_loss
from ..mixin import Revised_Mixin
from ..utils import model_card_template


@dataclass
class BiRefNetConfig:
    """Configuration for BiRefNet (Bilateral Reference Network) with Swin-L backbone."""

    image_size: int = 1024
    patch_size: int = 4
    embed_dim: int = 192
    depths: list = field(default_factory=lambda: [2, 2, 18, 2])
    num_heads: list = field(default_factory=lambda: [6, 12, 24, 48])
    window_size: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    dec_channels_inter: int = 64
    use_multi_scale_input: bool = True
    use_gradient_attention: bool = True
    use_image_patch_injection: bool = True


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class WindowAttention(nn.Module):
    relative_position_index: torch.Tensor

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W, mask_matrix):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, drop_path=None, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if drop_path is not None else 0.0,
            )
            for i in range(depth)
        ])

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        attn_mask = self._compute_mask(H, W, x.device)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)
        x_out = x

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_out, H, W, x_down, Wh, Ww
        else:
            return x_out, H, W, x, H, W

    def _compute_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, x.shape[2], Wh, Ww)
        return x, Wh, Ww


class SwinBackbone(nn.Module):
    def __init__(self, config: BiRefNetConfig):
        super().__init__()
        embed_dim = config.embed_dim
        depths = config.depths
        num_heads = config.num_heads
        window_size = config.window_size
        mlp_ratio = config.mlp_ratio
        drop_path_rate = config.drop_path_rate

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.out_indices = (0, 1, 2, 3)

        self.patch_embed = PatchEmbed(patch_size=config.patch_size, in_channels=3, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=True if (i_layer < self.num_layers - 1) else False,
            )
            self.layers.append(layer)

        for i_layer in self.out_indices:
            layer = nn.LayerNorm(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        x, Wh, Ww = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        ks = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * ks[0] * ks[1],
            kernel_size=ks, stride=stride, padding=padding, bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        assert self.offset_conv.bias is not None
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels, 1 * ks[0] * ks[1],
            kernel_size=ks, stride=stride, padding=padding, bias=True,
        )
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        assert self.modulator_conv.bias is not None
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ks,
            stride=stride, padding=padding, bias=bias,
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        x = deform_conv2d(
            input=x, offset=offset, weight=self.regular_conv.weight,
            bias=self.regular_conv.bias, padding=self.padding, mask=modulator, stride=self.stride,
        )
        return x


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = 256
        parallel_block_sizes = [1, 3, 7]

        self.aspp1 = _ASPPModuleDeformable(in_channels, inter_channels, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, inter_channels, conv_size, padding=conv_size // 2)
            for conv_size in parallel_block_sizes
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(inter_channels * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


def image2patches(image, patch_ref):
    grid_h = image.shape[-2] // patch_ref.shape[-2]
    grid_w = image.shape[-1] // patch_ref.shape[-1]
    B, C, H, W = image.shape
    h = H // grid_h
    w = W // grid_w
    x = image.view(B, C, grid_h, h, grid_w, w)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(B, C * grid_h * grid_w, h, w)
    return x


class Decoder(nn.Module):
    def __init__(self, channels: list[int], config: BiRefNetConfig):
        super().__init__()
        inter = config.dec_channels_inter

        ipt_cha = channels[0] // 8

        if config.use_image_patch_injection:
            # Input channels use image2patches: 3 * (image_size / ref_size)^2
            # For default 1024: x4=32x32->3*32*32=3072, x3=64->768, x2=128->192, x1=256->48, full=3
            ipt5_in = 3 * (config.image_size // (config.image_size // config.patch_size // 2**3))**2
            ipt4_in = 3 * (config.image_size // (config.image_size // config.patch_size // 2**2))**2
            ipt3_in = 3 * (config.image_size // (config.image_size // config.patch_size // 2**1))**2
            ipt2_in = 3 * (config.image_size // (config.image_size // config.patch_size))**2
            ipt1_in = 3
            self.ipt_blk5 = SimpleConvs(ipt5_in, ipt_cha, inter_channels=inter)
            self.ipt_blk4 = SimpleConvs(ipt4_in, ipt_cha, inter_channels=inter)
            self.ipt_blk3 = SimpleConvs(ipt3_in, channels[1] // 8, inter_channels=inter)
            self.ipt_blk2 = SimpleConvs(ipt2_in, channels[2] // 8, inter_channels=inter)
            self.ipt_blk1 = SimpleConvs(ipt1_in, channels[3] // 8, inter_channels=inter)

        self.decoder_block4 = BasicDecBlk(channels[0] + ipt_cha, channels[1], inter)
        self.decoder_block3 = BasicDecBlk(channels[1] + ipt_cha, channels[2], inter)
        self.decoder_block2 = BasicDecBlk(channels[2] + channels[1] // 8, channels[3], inter)
        self.decoder_block1 = BasicDecBlk(channels[3] + channels[2] // 8, channels[3] // 2, inter)

        self.conv_out1 = nn.Sequential(
            nn.Conv2d(channels[3] // 2 + channels[3] // 8, 1, 1, 1, 0)
        )

        self.lateral_block4 = BasicLatBlk(channels[1], channels[1])
        self.lateral_block3 = BasicLatBlk(channels[2], channels[2])
        self.lateral_block2 = BasicLatBlk(channels[3], channels[3])

        self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

        if config.use_gradient_attention:
            _N = 16
            self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
            self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
            self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
            self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
            self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
            self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
            self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
            self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
            self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

        self.use_gradient_attention = config.use_gradient_attention
        self.use_image_patch_injection = config.use_image_patch_injection

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        x, x1, x2, x3, x4 = features
        outs = []

        if self.use_image_patch_injection:
            patches = image2patches(x, patch_ref=x4)
            x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches, size=x4.shape[2:], mode="bilinear", align_corners=True))), 1)

        p4 = self.decoder_block4(x4)

        if self.use_gradient_attention:
            p4_gdt = self.gdt_convs_4(p4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            p4 = p4 * gdt_attn_4

        _p4 = F.interpolate(p4, size=x3.shape[2:], mode="bilinear", align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)

        if self.use_image_patch_injection:
            patches = image2patches(x, patch_ref=_p3)
            _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches, size=x3.shape[2:], mode="bilinear", align_corners=True))), 1)

        p3 = self.decoder_block3(_p3)

        if self.use_gradient_attention:
            p3_gdt = self.gdt_convs_3(p3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            p3 = p3 * gdt_attn_3

        _p3 = F.interpolate(p3, size=x2.shape[2:], mode="bilinear", align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)

        if self.use_image_patch_injection:
            patches = image2patches(x, patch_ref=_p2)
            _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches, size=x2.shape[2:], mode="bilinear", align_corners=True))), 1)

        p2 = self.decoder_block2(_p2)

        if self.use_gradient_attention:
            p2_gdt = self.gdt_convs_2(p2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            p2 = p2 * gdt_attn_2

        _p2 = F.interpolate(p2, size=x1.shape[2:], mode="bilinear", align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)

        if self.use_image_patch_injection:
            patches = image2patches(x, patch_ref=_p1)
            _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches, size=x1.shape[2:], mode="bilinear", align_corners=True))), 1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], mode="bilinear", align_corners=True)

        if self.use_image_patch_injection:
            patches = image2patches(x, patch_ref=_p1)
            _p1 = torch.cat((_p1, self.ipt_blk1(F.interpolate(patches, size=x.shape[2:], mode="bilinear", align_corners=True))), 1)

        p1_out = self.conv_out1(_p1)
        outs.append(p1_out)
        return outs


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
    """Bilateral Reference Network for high-resolution dichotomous image segmentation."""

    def __init__(self, config: Optional[BiRefNetConfig] = None):
        super().__init__()
        self.config = config or BiRefNetConfig()

        self.bb = SwinBackbone(self.config)

        base_channels = [self.config.embed_dim * (2**i) for i in range(len(self.config.depths))]

        if self.config.use_multi_scale_input:
            channels = [c * 2 for c in base_channels]
        else:
            channels = base_channels

        # channels = [C1, C2, C3, C4] shallow->deep
        # Decoder expects reversed: [C4, C3, C2, C1] deep->shallow
        dec_channels = list(reversed(channels))

        # Context: concat x1..x3 downsampled to x4 size, then cat with x4
        cxt_channels = channels[:-1]  # [C1, C2, C3]

        # Squeeze module: takes x4 with context concatenated
        squeeze_in = channels[-1] + sum(cxt_channels)
        self.squeeze_module = nn.Sequential(
            BasicDecBlk(squeeze_in, dec_channels[0], self.config.dec_channels_inter)
        )

        self.decoder = Decoder(dec_channels, self.config)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
        x = pixel_values
        x1, x2, x3, x4 = self.bb(x)

        if self.config.use_multi_scale_input:
            _, _, H, W = x.shape
            x_half = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=True)
            x1_, x2_, x3_, x4_ = self.bb(x_half)
            x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode="bilinear", align_corners=True)], dim=1)
            x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode="bilinear", align_corners=True)], dim=1)
            x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode="bilinear", align_corners=True)], dim=1)
            x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode="bilinear", align_corners=True)], dim=1)

        # Context aggregation
        x4 = torch.cat(
            (
                F.interpolate(x1, size=x4.shape[2:], mode="bilinear", align_corners=True),
                F.interpolate(x2, size=x4.shape[2:], mode="bilinear", align_corners=True),
                F.interpolate(x3, size=x4.shape[2:], mode="bilinear", align_corners=True),
                x4,
            ),
            dim=1,
        )

        x4 = self.squeeze_module(x4)

        scaled_preds = self.decoder([pixel_values, x1, x2, x3, x4])

        logits = scaled_preds[-1]
        if labels is not None:
            loss = birefnet_loss(scaled_preds, labels)
            return {"loss": loss, "logits": logits, "intermediate_logits": scaled_preds[:-1]}
        return {"logits": logits, "intermediate_logits": scaled_preds[:-1]}
