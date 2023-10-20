import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.deptwise_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.norm_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU6(inplace=True)

        self.pointwise_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm_pw = nn.BatchNorm2d(in_channels)
        self.act_pw = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise convolution
        x = self.deptwise_layer(x)
        x = self.norm_dw(x)
        x = self.act_dw(x)

        # Pointwise convolution
        x = self.pointwise_layer(x)
        x = self.norm_pw(x)
        x = self.act_pw(x)
        return x


def conv1x1(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0
):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


class FIMBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.dwconv = DWConv(in_channels, in_channels, kernel_size, stride, padding)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.avg_pool = nn.AvgPool2d(2)
        self.act = nn.Sigmoid()
        self.conv2 = conv1x1(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv(x)
        x = self.conv1(x)
        x = x * self.act(self.avg_pool(x))
        x = self.conv2(x)
        return x


class ReverseFIMBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.dwconv = DWConv(in_channels, in_channels, kernel_size, stride, padding)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.avg_pool = nn.AvgPool2d(2)
        self.act = nn.Sigmoid()
        self.conv2 = conv1x1(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x * self.act(self.avg_pool(x))
        x = self.conv2(x)
        x = x + self.dwconv(x)
        return x


class FIM(nn.Module):
    def __init__(self,):
        super().__init__()
        self.straight = FIMBlock()
        self.reverse = ReverseFIMBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.straight(x) + self.reverse(x)
        return x


class MHSA(nn.Module):
    def __init__(self, embed: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(embed, 3 * embed, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, embed, h, w = x.shape
        x = x.flatten(2).permute(1, 2).reshape(batch, h * w, self.heads, self.dim_head).permute(1, 2)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(attn)

        coarse_attn = attn.sum(-2)

        output = torch.matmul(attn, v)
        return coarse_attn, output


class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class RegionSelfAttention(nn.Module):
    def __init__(self, lmbd: int = 2):
        super().__init__()
        self.fim = FIM()
        self.lmbd = lmbd

    def forward(self, x: torch.Tensor, coarse_attn: torch.Tensor) -> torch.Tensor:
        batch_size, embed, h, w = x.shape
        x = self.fim(x)

        k_features = x.size(-1) // 4
        top_dots, top_inds = torch.topk(coarse_attn, k=k_features, dim=-1, sorted=False)
        top_inds *= self.lmbd

        region = torch.zeros_like(x)
        region[torch.arange(batch_size).view(3, 1), top_inds] = x[torch.arange(batch_size).view(3, 1, 1), top_inds]

        return x



class CoarseSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fim = FIM()
        self.mhsa = MHSA()
        self.upsample = Upsample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fim(x)
        coarse_attn, output = self.mhsa(x)
        output = self.upsample(output)
        return coarse_attn, output




class RegionSearchAttention(nn.Module):
    def __init__(self):
        self.csa = CoarseSelfAttention()
        self.rsa = RegionSelfAttention()
        self.dwconv = DWConv(in_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse_attn, coarse_output = self.csa(x)
        region_output = self.rsa(x, coarse_attn)
        x = coarse_output + region_output
        x = self.dwconv(x)
        return x


class RSABlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm()
        self.rsa = RegionSearchAttention()
        self.norm2 = nn.LayerNorm()
        self.mlp = nn.MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
