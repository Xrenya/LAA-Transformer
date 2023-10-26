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

"""
Test this upsample and downsample compared to papers' state: Conv2d and ConvTranspose2d
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
"""

class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=4,
            stride=2,
            padding=0
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
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
    def __init__(self, dim: int, head_dim: int = 4):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        
        self.upsample = Upsample(dim, dim)
        self.downsample = Downsample(dim, dim)
        
        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # downsample without increasing number of channels
        x = self.downsample(x)

        batch, channels, h, w = x.shape
        x = x.view(batch, channels // self.head_dim, self.head_dim, h * w).transpose(-2, -1)
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # batch, num_heads, patch, head_embed
        attn = torch.matmul(k, q.transpose(-2, -1))
        # batch, num_heads, patch, patch
        attn = self.softmax(attn)

        # [batch, num_heads, patch, patch] -> [batch, num_heads, patch]
        # patch score for batches for each head this propogate to the selection key regions by induces
        # which would be already splitted into number of head and head dim for further MHSA layers
        coarse_attn = torch.sum(attn, dim=-2)

        # since the number of features is decreasing the number of selected
        # topk should probably depend on the features size
        # from paper image the divisible of 4 was selected
        k_features = (h * w) // 4
        top_k = torch.topk(coarse_attn, k=k_features, dim=-1).indices

        output = torch.matmul(attn, v)
        output = output.permute(0, 1, 3, 2).reshape(batch, channels, h, w)
        output = self.upsample(output)
        return top_k, output




class RegionSelectionAttention(nn.Module):
    def __init__(self):
        self.course_attn = CoarseSelfAttention()
        self.topk_attn = TopkSelfAttention()
        self.dwconv = DWConv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_k, coarse_output = self.course_attn(x)
        region_output = self.rsa(coarse_output, top_k)
        x = coarse_output + region_output
        x = self.dwconv(x)
        return x


class RSABlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm()
        self.rsa = RegionSelectionAttention()
        self.norm2 = nn.LayerNorm()
        self.mlp = nn.MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TopkSelfAttention(nn.Module):
    def __init__(self, head_dim: int = 4):
        super().__init__()
        self.stride = 2
        self.kernel_size = 2
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        residual = x.clone()

        # [batch, embeddings, height, width]
        batch, embed, h, w = x.shape
        # [batch, num_heads, embeddings, height, width]
        residual = residual.view(batch, embed // self.head_dim, self.head_dim, h, w)
        print(residual.shape)
        x = x.view(batch, embed // self.head_dim, self.head_dim, h, w)
        # [batch, num_heads, embeddings, height // 2, width // 2, 2, 2]
        patches = x.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        patches = patches.reshape(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        # TODO remove loop with vectorize slicing
        batches = []
        for bat in range(batch):
            slices = []
            for i, inds in enumerate(top_k):
                patchify = []
                for idx in inds:
                    patchify.append(patches[bat, i, :, idx, :, :].unsqueeze(0).unsqueeze(1).unsqueeze(3))
                patchify = torch.cat(patchify, dim=3)
                slices.append(patchify)
            batches.append(torch.cat(slices, dim=1))
        # MHSA
        # [batch, num_heads, embeddings, top_k, top_k, 2, 2]
        # slices = torch.cat(slices, dim=1)
        batches = torch.cat(batches, dim=0)
        # batch, num_heads, embeddings, top_k, top_k, 2, 2
        # return batches
        batches = batches.flatten(3)
        batches = batches.transpose(-2, -1)

        q, k, v = self.to_qkv(batches).chunk(3, dim=-1)

        attn = torch.matmul(k, q.transpose(-2, -1))
        attn = self.softmax(attn)

        output = torch.matmul(attn, v)

        _, _, num_patches, _ = output.shape
        output = output.transpose(-2, -1)

        residual_patches = residual.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        residual_patches = residual_patches.reshape(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        output = output.view(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        for bat in range(batch):
            for i, inds in enumerate(top_k):
                for j, idx in enumerate(inds):
                    residual_patches[bat, i, :, idx, :, :] += output[bat, i, :, j, :, :]

        return residual_patches
    
class DWTResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        LL, LH, HL, HH = self.dwt(x)
        LL = self.conv(LL)
        LH = self.conv(LH)
        HL = self.conv(HL)
        return LL, LH, HL


class WFM(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.idwt = IDWT()
        self.upsample = Upsample(in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(
        self,
        ll: torch.tensor,
        lh: torch.tensor,
        hl: torch.tensor,
        x: torch.tensor
    ):
        inverse_feature, upsample_feature = x.chunk(2, dim=-1)
        inverse = self.idwt(ll, lh, hl, inverse_feature)
        x = self.upsample(upsample_feature)
        x = torch.cat([inverse, x])
        x = self.conv(x)
        return x
    
