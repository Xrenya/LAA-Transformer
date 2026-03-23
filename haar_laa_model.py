import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from torch.autograd import Function


# DWT / IDWT
class DWTFunction(Function):
    @staticmethod
    def forward(ctx, x, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1):
        ctx.save_for_backward(matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)
        L = torch.matmul(matrix_low_0, x)
        H = torch.matmul(matrix_high_0, x)
        LL = torch.matmul(L, matrix_low_1)
        LH = torch.matmul(L, matrix_high_1)
        HL = torch.matmul(H, matrix_low_1)
        HH = torch.matmul(H, matrix_high_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = ctx.saved_tensors
        grad_L = torch.add(
            torch.matmul(grad_LL, matrix_low_1.T),
            torch.matmul(grad_LH, matrix_high_1.T),
        )
        grad_H = torch.add(
            torch.matmul(grad_HL, matrix_low_1.T),
            torch.matmul(grad_HH, matrix_high_1.T),
        )
        grad_input = torch.add(
            torch.matmul(matrix_low_0.T, grad_L),
            torch.matmul(matrix_high_0.T, grad_H),
        )
        return grad_input, None, None, None, None


class IDWTFunction(Function):
    @staticmethod
    def forward(ctx, LL, LH, HL, HH, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1):
        ctx.save_for_backward(matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)
        L = torch.add(
            torch.matmul(LL, matrix_low_1.T),
            torch.matmul(LH, matrix_high_1.T),
        )
        H = torch.add(
            torch.matmul(HL, matrix_low_1.T),
            torch.matmul(HH, matrix_high_1.T),
        )
        output = torch.add(
            torch.matmul(matrix_low_0.T, L),
            torch.matmul(matrix_high_0.T, H),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = ctx.saved_tensors
        grad_L = torch.matmul(matrix_low_0, grad_output)
        grad_H = torch.matmul(matrix_high_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_low_1)
        grad_LH = torch.matmul(grad_L, matrix_high_1)
        grad_HL = torch.matmul(grad_H, matrix_low_1)
        grad_HH = torch.matmul(grad_H, matrix_high_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


def _generate_wavelet_matrices(band_low, band_high, height, width, device, dtype):
    """Shared utility to build the four transform matrices for DWT / IDWT"""
    band_length = len(band_low)
    max_side = max(height, width)
    half_side = max_side // 2

    mat_low = np.zeros((half_side, max_side + band_length - 2))
    mat_high = np.zeros((max_side - half_side, max_side + band_length - 2))

    end = None if band_length // 2 == 1 else -(band_length // 2) + 1

    idx = 0
    for i in range(half_side):
        for j in range(band_length):
            mat_low[i, idx + j] = band_low[j]
        idx += band_length

    idx = 0
    for i in range(max_side - half_side):
        for j in range(band_length):
            mat_high[i, idx + j] = band_high[j]
        idx += band_length

    mat_low_0 = mat_low[: height // 2, : height + band_length - 2]
    mat_low_1 = mat_low[: width // 2, : width + band_length - 2]

    mat_high_0 = mat_high[: height // 2, : height + band_length - 2]
    mat_high_1 = mat_high[: width // 2, : width + band_length - 2]

    mat_low_0 = mat_low_0[:, band_length // 2 - 1 : end]
    mat_low_1 = mat_low_1[:, band_length // 2 - 1 : end]
    mat_low_1 = mat_low_1.T

    mat_high_0 = mat_high_0[:, band_length // 2 - 1 : end]
    mat_high_1 = mat_high_1[:, band_length // 2 - 1 : end]
    mat_high_1 = mat_high_1.T

    matrix_low_0 = torch.tensor(mat_low_0, device=device, dtype=dtype)
    matrix_low_1 = torch.tensor(mat_low_1, device=device, dtype=dtype)
    matrix_high_0 = torch.tensor(mat_high_0, device=device, dtype=dtype)
    matrix_high_1 = torch.tensor(mat_high_1, device=device, dtype=dtype)

    return matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1


class DWT(nn.Module):
    """Discrete Wavelet Transform (Haar) - returns LL, LH, HL, HH"""

    def __init__(self):
        super().__init__()
        wavelet = pywt.Wavelet("haar")
        self.band_low = list(wavelet.rec_lo)
        self.band_high = list(wavelet.rec_hi)

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape
        mats = _generate_wavelet_matrices(
            self.band_low, self.band_high, h, w, x.device, x.dtype
        )
        return DWTFunction.apply(x, *mats)


class IDWT(nn.Module):
    """Inverse Discrete Wavelet Transform (Haar)"""

    def __init__(self):
        super().__init__()
        wavelet = pywt.Wavelet("haar")
        self.band_low = list(reversed(wavelet.rec_lo))
        self.band_high = list(reversed(wavelet.rec_hi))

    def forward(self, LL, LH, HL, HH):
        _, _, h, w = LL.shape
        mats = _generate_wavelet_matrices(
            self.band_low, self.band_high, h * 2, w * 2, LL.device, LL.dtype
        )
        return IDWTFunction.apply(LL, LH, HL, HH, *mats)


class DWConv(nn.Module):
    """Depthwise-separable convolution: DW‑Conv -> BN -> ReLU6 -> PW‑Conv -> BN -> ReLU6"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU6(inplace=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_dw(self.bn_dw(self.depthwise(x)))
        x = self.act_pw(self.bn_pw(self.pointwise(x)))
        return x


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, 1)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels * 2
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Embedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)

# FIM = Feature Interaction Module
class FIMBlock(nn.Module):
    """Straight-order FIM block: DWConv residual -> 1x1 -> channel-attention -> 1×1."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dwconv = DWConv(in_channels, in_channels, kernel_size, stride, padding)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.conv2 = conv1x1(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv(x)
        x = self.conv1(x)
        x = x * self.act(self.gap(x))  # broadcasts [B,C,1,1] → [B,C,H,W]
        x = self.conv2(x)
        return x


class ReverseFIMBlock(nn.Module):
    """Reverse‑order FIM block: 1x1 -> channel‑attention -> 1x1 -> DWConv residual"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.conv2 = conv1x1(out_channels, out_channels)
        self.dwconv = DWConv(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x * self.act(self.gap(x))
        x = self.conv2(x)
        x = x + self.dwconv(x)
        return x


class FIM(nn.Module):
    """Feature Interaction Module - parallel straight and reverse branches"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.straight = FIMBlock(in_channels, out_channels)
        self.reverse = ReverseFIMBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.straight(x) + self.reverse(x)


# Attention
class CoarseSelfAttention(nn.Module):
    """
    Downsample, MHSA, Upsample.
    Returns top‑k patch indices (per head) and the coarse attention output
    """

    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.downsample = Downsample(dim, dim)
        self.upsample = Upsample(dim, dim)

        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x_down = self.downsample(x)
        B, C, h, w = x_down.shape
        num_patches = h * w

        # [B, num_heads, num_patches, head_dim]
        tokens = (
            x_down.reshape(B, self.num_heads, self.head_dim, num_patches)
            .permute(0, 1, 3, 2)
        )

        q, k, v = self.to_qkv(tokens).chunk(3, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)  # [B, heads, P, P]

        coarse_score = attn.sum(dim=-2)  # [B, heads, P]

        k_features = max(1, num_patches // 4)
        top_k = torch.topk(coarse_score, k=k_features, dim=-1).indices  # [B, heads, k]

        output = torch.matmul(attn, v)  # [B, heads, P, head_dim]
        output = output.permute(0, 1, 3, 2).reshape(B, C, h, w)
        output = self.upsample(output)

        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return top_k, output


class TopkSelfAttention(nn.Module):
    """
    Select top‑k 2×2 patches (per head), run local MHSA, scatter back
    """

    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.patch_size = 2
        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        num_heads = C // self.head_dim
        ph, pw = H // ps, W // ps

        # [B, heads, head_dim, H, W] into patches [B, heads, head_dim, ph*pw, ps, ps]
        x_heads = x.reshape(B, num_heads, self.head_dim, H, W)
        patches = (
            x_heads
            .unfold(3, ps, ps)
            .unfold(4, ps, ps)  # [B, heads, hd, ph, pw, ps, ps]
            .reshape(B, num_heads, self.head_dim, ph * pw, ps, ps)
        )

        # top_k: [B, heads, k]
        k_sel = top_k.shape[-1]
        idx = top_k.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)  # [B, heads, 1, k, 1, 1]
        idx = idx.expand(B, num_heads, self.head_dim, k_sel, ps, ps)

        selected = torch.gather(patches, dim=3, index=idx)  # [B, heads, hd, k, ps, ps]

        tokens = selected.reshape(B, num_heads, self.head_dim, k_sel, ps * ps)
        tokens = tokens.permute(0, 1, 3, 4, 2)  # [B, heads, k, ps*ps, hd]
        tokens = tokens.reshape(B, num_heads, k_sel * ps * ps, self.head_dim)

        q, k_proj, v = self.to_qkv(tokens).chunk(3, dim=-1)
        attn = torch.matmul(q, k_proj.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out_tokens = torch.matmul(attn, v)  # [B, heads, k*ps*ps, hd]

        out_tokens = out_tokens.reshape(B, num_heads, k_sel, ps * ps, self.head_dim)
        out_tokens = out_tokens.permute(0, 1, 4, 2, 3)  # [B, heads, hd, k, ps*ps]
        out_tokens = out_tokens.reshape(B, num_heads, self.head_dim, k_sel, ps, ps)

        result_patches = torch.zeros_like(patches)
        result_patches.scatter_(3, idx, out_tokens)

        result = result_patches.reshape(B, num_heads, self.head_dim, ph, pw, ps, ps)
        result = (
            result.permute(0, 1, 2, 3, 5, 4, 6)  # [B, heads, hd, ph, ps, pw, ps]
            .reshape(B, C, H, W)
        )
        return result


class RegionSelectionAttention(nn.Module):
    """Coarse attention - top‑k fine attention - DWConv feed‑forward."""

    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.coarse_attn = CoarseSelfAttention(dim, head_dim)
        self.topk_attn = TopkSelfAttention(head_dim)
        self.dwconv = DWConv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_k, coarse_out = self.coarse_attn(x)
        region_out = self.topk_attn(coarse_out, top_k)
        x = coarse_out + region_out
        x = self.dwconv(x)
        return x

# RSA Block
class RSABlock(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.rsa = RegionSelectionAttention(dim, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # RSA
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        normed = self.norm1(x_flat)
        normed = normed.transpose(1, 2).reshape(B, C, H, W)  # back to spatial
        x = x + self.rsa(normed)

        # MLP
        x_flat = x.flatten(2).transpose(1, 2)
        normed = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(normed)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


# DWT
class DWTResidual(nn.Module):
    """DWT"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        LL = self.conv(LL)
        LH = self.conv(LH)
        HL = self.conv(HL)
        return LL, LH, HL


class WFM(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.idwt = IDWT()
        half = in_channels // 2
        self.upsample = Upsample(half, half)
        self.conv = nn.Conv2d(in_channels, half, 1)

    def forward(self, ll, lh, hl, x):
        inverse_feature, upsample_feature = x.chunk(2, dim=1)

        if inverse_feature.shape[-2:] != ll.shape[-2:]:
            inverse_feature = F.interpolate(
                inverse_feature,
                size=ll.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        inverse = self.idwt(ll, lh, hl, inverse_feature)

        up = self.upsample(upsample_feature)
        if up.shape[-2:] != inverse.shape[-2:]:
            up = F.interpolate(up, size=inverse.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([inverse, up], dim=1)
        x = self.conv(x)
        return x

# LAA Model
class LAAModel(nn.Module):
    """
    LAA‑Net: U-shaped encoder–decoder with
      - RSA (Region‑Selection Attention) blocks
      - DWT skip connections (encoder/decoder)
      - WFM wavelet feature merge in decoder
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 3,
        depths: list = None,
        head_dim: int = 64,
        image_size: int = 512,
    ):
        super().__init__()
        if depths is None:
            depths = [64, 128, 256, 512]

        self.conv_in = nn.Conv2d(in_channels, depths[0], 3, padding=1)
        self.act_in = nn.LeakyReLU(inplace=True)
        self.embed = Embedding(depths[0], depths[0], patch_size)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.dwt_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, d in enumerate(depths):
            self.enc_blocks.append(RSABlock(d, head_dim))
            if i < len(depths) - 1:
                self.dwt_blocks.append(DWTResidual(d))
                self.downsamples.append(Downsample(d, depths[i + 1]))
            else:
                self.dwt_blocks.append(None)
                self.downsamples.append(None)

        # Bottlenec
        self.bottleneck = RSABlock(depths[-1], head_dim)
        self.bottleneck_cat = nn.Conv2d(depths[-1] * 2, depths[-1] * 2, 1)
        self.bottleneck_act = nn.LeakyReLU(inplace=True)

        # Decoder
        rev = list(reversed(depths))
        self.dec_blocks = nn.ModuleList()
        self.wfm_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, d in enumerate(rev):
            dec_ch = d * 2
            self.dec_blocks.append(RSABlock(dec_ch, head_dim))
            if i < len(rev) - 1:
                self.wfm_blocks.append(WFM(dec_ch))
                next_d = rev[i + 1] * 2
                self.upsamples.append(
                    nn.Sequential(
                        Upsample(d, d),
                        nn.Conv2d(d, next_d, 1),
                    )
                    if d != next_d
                    else Upsample(d, next_d)
                )
            else:
                self.wfm_blocks.append(None)
                self.upsamples.append(None)

        # Head
        self.upsample_out = nn.ConvTranspose2d(
            rev[-1] * 2, depths[0], patch_size, stride=patch_size,
        )
        self.conv_out = nn.Conv2d(depths[0], in_channels, 3, padding=1)
        self.act_out = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()

        x = self.act_in(self.conv_in(x))
        x = self.embed(x)

        # Encoder
        dwt_skips = [] # will store (LL, LH, HL) tuples
        for enc, down, dwt in zip(self.enc_blocks, self.downsamples, self.dwt_blocks):
            x = enc(x)
            if dwt is not None:
                dwt_skips.append(dwt(x))
            if down is not None:
                x = down(x)

        dwt_skips.reverse() # decoder order

        # Bottleneck
        x = torch.cat([self.bottleneck(x), x], dim=1)
        x = self.bottleneck_act(self.bottleneck_cat(x))

        # Decode
        for i, (dec, wfm, up) in enumerate(
            zip(self.dec_blocks, self.wfm_blocks, self.upsamples)
        ):
            x = dec(x)
            if wfm is not None and i < len(dwt_skips):
                ll, lh, hl = dwt_skips[i]
                x = wfm(ll, lh, hl, x)
            if up is not None:
                x = up(x)

        # Head
        x = self.upsample_out(x)
        x = self.act_out(self.conv_out(x))

        if x.shape[-2:] != residual.shape[-2:]:
            x = F.interpolate(x, size=residual.shape[-2:], mode="bilinear", align_corners=False)

        return residual + x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LAAModel(
        patch_size=2,
        in_channels=3,
        depths=[64, 128, 256, 512],
        head_dim=64,
        image_size=256,
    ).to(device)

    x = torch.randn(1, 3, 256, 256, device=device)  # 512x512 out of memory probably in CoarseSelfAttention - possible solution is downsample features, is it how it dones in the paper? :\\ 

    # count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f} M")

    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"
    print("Works")
