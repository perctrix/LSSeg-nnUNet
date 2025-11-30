import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import override, Literal
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

from mipcandy import SegmentationTrainer, LayerT
from mipcandy.training import TrainerToolbox


class SKA(nn.Module):
    """
    SKA: Small Kernel Aggregation
    Supports:
      2D: x -> (B,C,H,W) or (C,H,W); w -> (B,C_w, ks^2, H, W)
      3D: x -> (B,C,D,H,W) or (C,D,H,W); w -> (B,C_w, ks^3, D, H, W)
    Notes:
      - For 3D, this class uses standard PyTorch format BCDHW.
      - stride=1, dilation=1; padding=ks//2
      - groups = C // C_w
    """

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch = True
        elif x.dim() == 3:
            pass
        elif x.dim() == 4:
            pass
        elif x.dim() == 5:
            pass
        else:
            raise ValueError(f"Unsupported x.dim()={x.dim()}")

        B, C = x.shape[:2]
        is_1d = (x.dim() == 3)
        is_3d = (x.dim() == 5)

        ks_pow = w.shape[2]
        if is_1d:
            ks = ks_pow
        elif is_3d:
            ks = round(ks_pow ** (1/3))
            if ks ** 3 != ks_pow:
                raise ValueError(f"w.shape[2]={ks_pow} is not a perfect cube")
        else:
            ks = int(math.isqrt(ks_pow))
            if ks * ks != ks_pow:
                raise ValueError(f"w.shape[2]={ks_pow} is not a perfect square")

        pad = ks // 2
        C_w = w.shape[1]
        if C % C_w != 0:
            raise ValueError(f"C={C} is not divisible by C_w={C_w}")
        groups = C // C_w

        if is_1d:
            _, _, L = x.shape
            x_unfold = F.unfold(x.unsqueeze(-1), kernel_size=(ks, 1), padding=(pad, 0))
            x_unfold = x_unfold.view(B, C, ks, L)

            x_unfold = x_unfold.view(B, groups, C_w, ks, L).transpose(1, 2).contiguous()

            w = w.unsqueeze(2)

            out = (x_unfold * w).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, C, L)

            if added_batch:
                out = out.squeeze(0)
            return out

        elif not is_3d:
            _, _, H, W = x.shape
            x_unfold = F.unfold(x, kernel_size=ks, padding=pad)
            x_unfold = x_unfold.view(B, C, ks*ks, H, W)

            x_unfold = x_unfold.view(B, groups, C_w, ks*ks, H, W).transpose(1, 2).contiguous()

            w = w.unsqueeze(2)

            out = (x_unfold * w).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, C, H, W)

            if added_batch:
                out = out.squeeze(0)
            return out

        else:
            B, C, D, H, W = x.shape

            x_pad = F.pad(x, (pad, pad, pad, pad, pad, pad))

            patches = (
                x_pad.unfold(2, ks, 1)
                     .unfold(3, ks, 1)
                     .unfold(4, ks, 1)
            ).contiguous()

            Bp, Cp, D_out, H_out, W_out, *_ = patches.shape
            assert Bp==B and Cp==C
            ks_cubed = ks**3

            patches = patches.permute(0,1,5,6,7,2,3,4).contiguous()
            patches = patches.view(B, C, ks_cubed, D_out, H_out, W_out)

            patches = patches.view(B, groups, C_w, ks_cubed, D_out, H_out, W_out).transpose(1, 2).contiguous()

            if w.dim() != 6:
                raise ValueError(f"For 3D, expected w.dim()==6, got {w.dim()}")
            if w.shape[0] != B or w.shape[1] != C_w or w.shape[2] != ks_cubed:
                raise ValueError(f"w shape mismatch: expected (B,{C_w},{ks_cubed},D,H,W), got {tuple(w.shape)}")
            w = w.unsqueeze(2)

            out = (patches * w).sum(dim=3)
            out = out.transpose(1, 2).contiguous().view(B, C, D_out, H_out, W_out)

            if added_batch:
                out = out.squeeze(0)
            return out


class Residual(nn.Module):

    def __init__(self, fn: nn.Module, *, drop: float = 0.0) -> None:
        super().__init__()
        self.fn: nn.Module = fn
        self.drop: float = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            mask_shape = [x.size(0)] + [1] * (x.dim() - 1)
            mask = torch.rand(*mask_shape, device=x.device).ge_(self.drop).div(1 - self.drop)
            return x + self.fn(x) * mask
        return x + self.fn(x)


class FFN(nn.Module):

    def __init__(self, dim: int, *, expansion: int = 2, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                 act: LayerT = LayerT(nn.ReLU)) -> None:
        super().__init__()
        hidden = dim * expansion
        self.conv1: nn.Module = conv.assemble(in_channels=dim, out_channels=hidden, kernel_size=1)
        self.norm1: nn.Module = norm.assemble(in_ch=hidden)
        self.act: nn.Module = act.assemble()
        self.conv2: nn.Module = conv.assemble(in_channels=hidden, out_channels=dim, kernel_size=1)
        self.norm2: nn.Module = norm.assemble(in_ch=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class SqueezeExcite(nn.Module):

    def __init__(self, dim: int, *, ratio: float = 0.25, num_dims: Literal[1, 2, 3] = 2,
                 conv: LayerT = LayerT(nn.Conv2d)) -> None:
        super().__init__()
        self.num_dims: int = num_dims
        hidden = max(1, int(dim * ratio))
        self.fc1: nn.Module = conv.assemble(in_channels=dim, out_channels=hidden, kernel_size=1)
        self.act: nn.Module = nn.ReLU(inplace=True)
        self.fc2: nn.Module = conv.assemble(in_channels=hidden, out_channels=dim, kernel_size=1)
        self.gate: nn.Module = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_dims == 1:
            scale = F.adaptive_avg_pool1d(x, 1)
        elif self.num_dims == 2:
            scale = F.adaptive_avg_pool2d(x, 1)
        else:
            scale = F.adaptive_avg_pool3d(x, 1)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class RepVGGDW(nn.Module):

    def __init__(self, dim: int, *, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch")) -> None:
        super().__init__()
        self.conv3: nn.Module = conv.assemble(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm3: nn.Module = norm.assemble(in_ch=dim)
        self.conv1: nn.Module = conv.assemble(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.norm1: nn.Module = norm.assemble(in_ch=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm3(self.conv3(x)) + self.norm1(self.conv1(x)) + x


class LKP(nn.Module):
    """
    LKP: Large Kernel Perception
    """

    def __init__(self, dim: int, *, lks: int = 7, sks: int = 3, groups: int = 8, num_dims: Literal[1, 2, 3] = 2,
                 conv: LayerT = LayerT(nn.Conv2d), norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                 act: LayerT = LayerT(nn.ReLU)) -> None:
        super().__init__()
        self.num_dims: int = num_dims
        self.sks: int = sks
        self.groups: int = groups
        self.dim: int = dim

        ks_pow = sks ** num_dims

        self.conv1: nn.Module = conv.assemble(in_channels=dim, out_channels=dim // 2, kernel_size=1, bias=False)
        self.norm1: nn.Module = norm.assemble(in_ch=dim // 2)
        self.act1: nn.Module = act.assemble()

        self.conv2: nn.Module = conv.assemble(in_channels=dim // 2, out_channels=dim // 2, kernel_size=lks, padding=lks // 2, groups=dim // 2, bias=False)
        self.norm2: nn.Module = norm.assemble(in_ch=dim // 2)
        self.act2: nn.Module = act.assemble()

        self.conv3: nn.Module = conv.assemble(in_channels=dim // 2, out_channels=dim // 2, kernel_size=1, bias=False)
        self.norm3: nn.Module = norm.assemble(in_ch=dim // 2)
        self.act3: nn.Module = act.assemble()

        self.conv4: nn.Module = conv.assemble(in_channels=dim // 2, out_channels=ks_pow * dim // groups, kernel_size=1)
        self.norm4: nn.Module = nn.GroupNorm(dim // groups, ks_pow * dim // groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        w = self.norm4(self.conv4(x))

        ks_pow = self.sks ** self.num_dims
        if self.num_dims == 1:
            b, _, length = w.size()
            w = w.view(b, self.dim // self.groups, ks_pow, length)
        elif self.num_dims == 2:
            b, _, h, width = w.size()
            w = w.view(b, self.dim // self.groups, ks_pow, h, width)
        else:
            b, _, d, h, width = w.size()
            w = w.view(b, self.dim // self.groups, ks_pow, d, h, width)
        return w


class LSConv(nn.Module):

    def __init__(self, dim: int, *, num_dims: Literal[1, 2, 3] = 2, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch")) -> None:
        super().__init__()
        self.lkp: nn.Module = LKP(dim, lks=7, sks=3, groups=8, num_dims=num_dims, conv=conv, norm=norm)
        self.ska: nn.Module = SKA()
        self.norm: nn.Module = norm.assemble(in_ch=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lkp(x)
        out = self.ska(x, w)
        return self.norm(out) + x


class Attention(nn.Module):

    def __init__(self, dim: int, *, num_heads: int = 8, head_dim: int = 16, num_dims: Literal[1, 2, 3] = 2,
                 conv: LayerT = LayerT(nn.Conv2d), norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch")) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.scale: float = head_dim ** -0.5
        self.num_dims: int = num_dims

        inner_dim = head_dim * num_heads
        self.qkv: nn.Module = conv.assemble(in_channels=dim, out_channels=inner_dim * 3, kernel_size=1)
        self.dw: nn.Module = conv.assemble(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, padding=1, groups=inner_dim)
        self.proj: nn.Module = nn.Sequential(
            nn.ReLU(),
            conv.assemble(in_channels=inner_dim, out_channels=dim, kernel_size=1, bias=False),
            norm.assemble(in_ch=dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_dims == 1:
            B, _, L = x.shape
            N = L
            spatial_shape = (L,)
        elif self.num_dims == 2:
            B, _, H, W = x.shape
            N = H * W
            spatial_shape = (H, W)
        else:
            B, _, D, H, W = x.shape
            N = D * H * W
            spatial_shape = (D, H, W)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = self.dw(q)

        q = q.view(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, -1, *spatial_shape)
        return self.proj(out)


class Block(nn.Module):

    def __init__(self, dim: int, *, stage: int = 0, depth: int = 0, num_heads: int = 8,
                 head_dim: int = 16, num_dims: Literal[1, 2, 3] = 2, conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch")) -> None:
        super().__init__()

        if depth % 2 == 0:
            self.mixer: nn.Module = RepVGGDW(dim, conv=conv, norm=norm)
            self.se: nn.Module = SqueezeExcite(dim, ratio=0.25, num_dims=num_dims, conv=conv)
        else:
            self.se: nn.Module = nn.Identity()
            if stage == 3:
                self.mixer: nn.Module = Residual(Attention(dim, num_heads=num_heads, head_dim=head_dim,
                                                           num_dims=num_dims, conv=conv, norm=norm))
            else:
                self.mixer: nn.Module = LSConv(dim, num_dims=num_dims, conv=conv, norm=norm)

        self.ffn: nn.Module = Residual(FFN(dim, expansion=2, conv=conv, norm=norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.se(x)
        x = self.ffn(x)
        return x


class LSDecoder(nn.Module):
    """
    Large-Small Decoder: LSConv for content-adaptive feature reconstruction.

    Structure:
    1. Upsample
    2. Channel adjustment convolution
    3. Skip connection concatenation
    4. LSConv block - dynamic large-small kernel fusion + FFN refinement

    Args:
        in_ch: Input channels (from previous decoder layer)
        out_ch: Output channels (current layer feature dimension)
        num_dims: 2D or 3D
        conv: Convolution layer configuration
        norm: Normalization layer configuration
        act: Activation function configuration
    """

    def __init__(self, in_ch: int, out_ch: int, *,
                 num_dims: Literal[1, 2, 3] = 2,
                 conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                 act: LayerT = LayerT(nn.ReLU)) -> None:
        super().__init__()
        if num_dims == 1:
            upsample_mode = 'linear'
        elif num_dims == 2:
            upsample_mode = 'bilinear'
        else:
            upsample_mode = 'trilinear'

        self.up: nn.Module = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        self.conv_up: nn.Module = conv.assemble(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
        self.norm_up: nn.Module = norm.assemble(in_ch=out_ch)
        self.act_up: nn.Module = act.assemble()

        self.pre_fusion: nn.Module = nn.Sequential(
            conv.assemble(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=1, bias=False),
            norm.assemble(in_ch=out_ch),
            act.assemble()
        )
        self.fusion: nn.Module = nn.Sequential(
            LSConv(out_ch, num_dims=num_dims, conv=conv, norm=norm),
            Residual(FFN(out_ch, expansion=2, conv=conv, norm=norm, act=act))
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.act_up(self.norm_up(self.conv_up(x)))

        x = torch.cat([x, skip], dim=1)

        x = self.pre_fusion(x)
        x = self.fusion(x)
        return x


class LSNetSegD3(nn.Module):
    """
    LSNet-Seg with D3 configuration (Full LSConv decoder).

    Architecture:
    - Encoder: LSNet with LSConv + Attention (stage 3 only)
    - Decoder: Full LSConv decoder (all three layers use LSConv)
    - Skip connection: Simple concatenation
    - Deep supervision: Optional

    This is the best configuration from decoder ablation experiments:
    - Dice: 0.9696
    - Boundary IoU: 0.4689
    - Parameters: 11.71M (variant "t")
    """

    def __init__(self, in_ch: int, num_classes: int, *,
                 hidden_chs: tuple[int, int, int, int] = (64, 128, 256, 384),
                 depths: tuple[int, int, int, int] = (0, 2, 8, 10),
                 num_heads: tuple[int, int, int, int] = (3, 3, 3, 4),
                 head_dims: tuple[int, int, int, int] = (16, 16, 16, 16),
                 num_dims: Literal[1, 2, 3] = 2,
                 conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                 enable_deep_supervision: bool = False) -> None:
        super().__init__()

        self.num_dims: int = num_dims
        if num_dims == 1:
            upsample_mode = 'linear'
        elif num_dims == 2:
            upsample_mode = 'bilinear'
        else:
            upsample_mode = 'trilinear'

        self.stem: nn.Module = nn.Sequential(
            conv.assemble(in_channels=in_ch, out_channels=hidden_chs[0] // 4, kernel_size=3, stride=2, padding=1, bias=False),
            norm.assemble(in_ch=hidden_chs[0] // 4),
            nn.ReLU(),
            conv.assemble(in_channels=hidden_chs[0] // 4, out_channels=hidden_chs[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            norm.assemble(in_ch=hidden_chs[0] // 2),
            nn.ReLU(),
            conv.assemble(in_channels=hidden_chs[0] // 2, out_channels=hidden_chs[0], kernel_size=3, stride=2, padding=1, bias=False),
            norm.assemble(in_ch=hidden_chs[0])
        )

        self.stages: nn.ModuleList = nn.ModuleList()
        for i in range(len(hidden_chs)):
            stage = nn.Sequential()

            if i > 0:
                stage.append(conv.assemble(in_channels=hidden_chs[i-1], out_channels=hidden_chs[i-1], kernel_size=3, stride=2, padding=1, groups=hidden_chs[i-1], bias=False))
                stage.append(norm.assemble(in_ch=hidden_chs[i-1]))
                stage.append(conv.assemble(in_channels=hidden_chs[i-1], out_channels=hidden_chs[i], kernel_size=1, bias=False))
                stage.append(norm.assemble(in_ch=hidden_chs[i]))

            for d in range(depths[i]):
                stage.append(Block(hidden_chs[i], stage=i, depth=d, num_heads=num_heads[i],
                                 head_dim=head_dims[i], num_dims=num_dims, conv=conv, norm=norm))

            self.stages.append(stage)

        self.enable_deep_supervision: bool = enable_deep_supervision

        self.decoder1: nn.Module = LSDecoder(
            hidden_chs[3], hidden_chs[2],
            num_dims=num_dims, conv=conv, norm=norm
        )
        self.decoder2: nn.Module = LSDecoder(
            hidden_chs[2], hidden_chs[1],
            num_dims=num_dims, conv=conv, norm=norm
        )
        self.decoder3: nn.Module = LSDecoder(
            hidden_chs[1], hidden_chs[0],
            num_dims=num_dims, conv=conv, norm=norm
        )

        self.final_up: nn.Module = nn.Upsample(scale_factor=8, mode=upsample_mode, align_corners=True)
        self.out: nn.Module = conv.assemble(in_channels=hidden_chs[0], out_channels=num_classes, kernel_size=1)

        if enable_deep_supervision:
            self.ds_out1: nn.Module = conv.assemble(in_channels=hidden_chs[2], out_channels=num_classes, kernel_size=1)
            self.ds_up1: nn.Module = nn.Upsample(scale_factor=32, mode=upsample_mode, align_corners=True)
            self.ds_out2: nn.Module = conv.assemble(in_channels=hidden_chs[1], out_channels=num_classes, kernel_size=1)
            self.ds_up2: nn.Module = nn.Upsample(scale_factor=16, mode=upsample_mode, align_corners=True)
            self.ds_out3: nn.Module = conv.assemble(in_channels=hidden_chs[0], out_channels=num_classes, kernel_size=1)
            self.ds_up3: nn.Module = nn.Upsample(scale_factor=8, mode=upsample_mode, align_corners=True)

        self.num_classes: int = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.stem(x)

        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)

        d1 = self.decoder1(skips[3], skips[2])
        d2 = self.decoder2(d1, skips[1])
        d3 = self.decoder3(d2, skips[0])

        if self.training and self.enable_deep_supervision:
            ds1 = self.ds_up1(self.ds_out1(d1))
            ds2 = self.ds_up2(self.ds_out2(d2))
            ds3 = self.ds_up3(self.ds_out3(d3))
            main_out = self.final_up(self.out(d3))
            return [main_out, ds1, ds2, ds3]
        else:
            x = self.final_up(d3)
            x = self.out(x)
            return x


def compute_boundary_iou(pred: torch.Tensor, gt: torch.Tensor, *, width: int = 2, threshold: float = 0.5) -> float:
    """
    Compute boundary IoU.

    Args:
        pred: Prediction mask, shape (H, W) or (B, 1, H, W), range [0, 1]
        gt: Ground truth mask, shape (H, W) or (B, 1, H, W), range {0, 1}
        width: Boundary width in pixels
        threshold: Prediction binarization threshold

    Returns:
        boundary_iou: IoU value in boundary region
    """
    if pred.dim() == 4:
        pred = pred.squeeze(0).squeeze(0)
    if gt.dim() == 4:
        gt = gt.squeeze(0).squeeze(0)

    pred_np = (pred.detach().cpu().numpy() > threshold).astype(np.uint8)
    gt_np = gt.detach().cpu().numpy().astype(np.uint8)

    dist_fg = distance_transform_edt(gt_np == 0)
    dist_bg = distance_transform_edt(gt_np == 1)

    boundary_mask = (dist_fg <= width) & (dist_bg <= width)

    if boundary_mask.sum() == 0:
        return 0.0

    pred_boundary = pred_np[boundary_mask]
    gt_boundary = gt_np[boundary_mask]

    intersection = (pred_boundary & gt_boundary).sum()
    union = (pred_boundary | gt_boundary).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def evaluate_segmentation(pred: torch.Tensor, gt: torch.Tensor, *,
                          threshold: float = 0.5,
                          boundary_width: int = 2) -> dict[str, float]:
    """
    Comprehensive segmentation evaluation.

    Args:
        pred: Prediction mask, shape (H, W) or (B, 1, H, W), range [0, 1]
        gt: Ground truth mask, shape (H, W) or (B, 1, H, W), range {0, 1}
        threshold: Prediction binarization threshold
        boundary_width: Boundary IoU width in pixels

    Returns:
        metrics: Dictionary containing evaluation metrics
            - boundary_iou: Boundary IoU
    """
    metrics = {
        'boundary_iou': compute_boundary_iou(pred, gt, width=boundary_width, threshold=threshold),
    }

    return metrics


class LSNetSegD3Trainer(SegmentationTrainer):
    """
    LSNet-Seg Trainer with D3 configuration (Full LSConv decoder).

    Configuration:
    - Encoder: LSConv + Attention (stage 3 only)
    - Decoder: Full LSConv (all three decoder layers use LSConv)
    - Skip connection: Simple concatenation
    - Deep supervision: Enabled

    This is the best configuration from decoder ablation experiments:
    - Dice: 0.9696
    - Boundary IoU: 0.4689
    - Parameters: 11.71M
    """

    variant: Literal["t", "s", "b"] = "t"
    deep_supervision: bool = True

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return build_lsnet_seg_d3(
            in_ch=example_shape[0],
            num_classes=self.num_classes,
            variant=self.variant,
            num_dims=self.num_dims,
            enable_deep_supervision=self.deep_supervision
        )

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor,
                toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
        outputs = toolbox.model(images)

        if self.deep_supervision:
            total_loss = 0
            for output in outputs:
                loss, _ = toolbox.criterion(output, labels)
                total_loss += loss
            total_loss = total_loss / len(outputs)

            final_output = outputs[0]
            _, metrics = toolbox.criterion(final_output, labels)
        else:
            total_loss, metrics = toolbox.criterion(outputs, labels)

        total_loss.backward()
        return total_loss.item(), metrics

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor,
                     toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        outputs = (toolbox.ema if toolbox.ema else toolbox.model)(image)

        loss, metrics = toolbox.criterion(outputs, label)

        pred = torch.sigmoid(outputs.squeeze(0))
        boundary_metrics = evaluate_segmentation(pred, label.squeeze(0),
                                                threshold=0.5, boundary_width=2)

        metrics['boundary_iou'] = boundary_metrics['boundary_iou']

        return -loss.item(), metrics, outputs.squeeze(0)


def build_lsnet_seg_d3(in_ch: int, num_classes: int, *,
                      variant: Literal["t", "s", "b"] = "t",
                      num_dims: Literal[1, 2, 3] = 2,
                      conv: LayerT | None = None,
                      norm: LayerT = LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                      enable_deep_supervision: bool = True) -> nn.Module:
    """
    Build LSNet-Seg with D3 configuration (Full LSConv decoder).

    Args:
        in_ch: Number of input channels
        num_classes: Number of output classes
        variant: Model variant ("t" for tiny, "s" for small, "b" for base)
        num_dims: Number of spatial dimensions (2 or 3)
        conv: Convolution layer configuration (optional)
        norm: Normalization layer configuration
        enable_deep_supervision: Whether to enable deep supervision

    Returns:
        model: LSNet-Seg model with D3 configuration

    Architecture:
        - Encoder: LSNet with LSConv + Attention (stage 3)
        - Decoder: Full LSConv decoder (True, True, True)
        - Skip: Simple concatenation
        - Deep supervision: Enabled (default)

    Performance (PH2 dataset):
        - Soft Dice: 0.9696
        - Boundary IoU: 0.4689
        - Parameters: 11.71M (variant "t")
    """
    if conv is None:
        if num_dims == 1:
            conv = LayerT(nn.Conv1d)
        elif num_dims == 2:
            conv = LayerT(nn.Conv2d)
        else:
            conv = LayerT(nn.Conv3d)

    configs = {
        "t": {"hidden_chs": (64, 128, 256, 384), "depths": (0, 2, 8, 10)},
        "s": {"hidden_chs": (96, 192, 320, 448), "depths": (1, 2, 8, 10)},
        "b": {"hidden_chs": (128, 256, 384, 512), "depths": (4, 6, 8, 10)},
    }
    config = configs[variant]

    return LSNetSegD3(
        in_ch, num_classes,
        hidden_chs=config["hidden_chs"],
        depths=config["depths"],
        num_heads=(3, 3, 3, 4),
        head_dims=(16, 16, 16, 16),
        num_dims=num_dims,
        conv=conv,
        norm=norm,
        enable_deep_supervision=enable_deep_supervision
    )


if __name__ == "__main__":
    from mipcandy import sanity_check

    lsnet_seg_2d = build_lsnet_seg_d3(in_ch=2, num_classes=3, variant="b", num_dims=2,
                                      conv=LayerT(nn.Conv2d), norm=LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                                      enable_deep_supervision=True)
    results_2d = sanity_check(lsnet_seg_2d, input_shape=(2, 128, 128))
    print(results_2d)

    lsnet_seg_3d = build_lsnet_seg_d3(in_ch=1, num_classes=2, variant="b", num_dims=3,
                                      conv=LayerT(nn.Conv3d), norm=LayerT(nn.GroupNorm, num_groups=8, num_channels="in_ch"),
                                      enable_deep_supervision=False)
    results_3d = sanity_check(lsnet_seg_3d, input_shape=(1, 64, 64, 64))
    print(results_3d)
    
    lsnet_1d = build_lsnet_seg_d3(in_ch=1, num_classes=2, variant="b", num_dims=1,
                                  conv=LayerT(nn.Conv1d), norm=LayerT(nn.InstanceNorm1d, num_features="in_ch"),
                                  enable_deep_supervision=False)
    results_1d = sanity_check(lsnet_1d, input_shape=(1, 128))
    print(results_1d)
