import torch
import torch.nn as nn
import pywt
import numpy as np
from torch.autograd import Function


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        wavelet = pywt.Wavelet("haar")
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        self.band_length = len(self.band_low)

    def generate_mat(self, height: int, width: int, device, dtype):
        max_side = max(height, width)
        half_side = max_side // 2
        mat_height = np.zeros((half_side, max_side + self.band_length - 2))
        mat_width = np.zeros((max_side - half_side, max_side + self.band_length - 2))
        end = None if self.band_length // 2 == 1 else -self.band_length // 2 + 1

        idx = 0
        for i in range(half_side):
            for j in range(self.band_length):
                mat_height[i, idx + j] = self.band_low[j]
            idx += self.band_length
        mat_height_0 = mat_height[0:height // 2, 0:height + self.band_length - 2]
        mat_height_1 = mat_height[0:width // 2, 0:width + self.band_length - 2]

        idx = 0
        for i in range(max_side - half_side):
            for j in range(self.band_length):
                mat_width[i, idx + j] = self.band_high[j]
            idx += self.band_length
        mat_width_0 = mat_width[0:height // 2, 0:height + self.band_length - 2]
        mat_width_1 = mat_width[0:width // 2, 0:width + self.band_length - 2]

        mat_height_0 = mat_height_0[:, self.band_length // 2 - 1:end]
        mat_height_1 = mat_height_1[:, self.band_length // 2 - 1:end]
        mat_height_1 = np.transpose(mat_height_1)

        mat_width_0 = mat_width_0[:, self.band_length // 2 - 1:end]
        mat_width_1 = mat_width_1[:, self.band_length // 2 - 1:end]
        mat_width_1 = np.transpose(mat_width_1)

        self.matrix_low_0 = torch.tensor(mat_height_0, device=device, dtype=dtype)
        self.matrix_low_1 = torch.tensor(mat_height_1, device=device, dtype=dtype)
        self.matrix_high_0 = torch.tensor(mat_width_0, device=device, dtype=dtype)
        self.matrix_high_1 = torch.tensor(mat_width_1, device=device, dtype=dtype)

    def forward(self, x: torch.tensor) -> torch.tensor:
        device = x.device
        batch, channels, height, width = x.shape
        self.generate_mat(height, width, x.device, x.dtype)
        return DWTFunction.apply(x, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWTFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.tensor,
        matrix_low_0: torch.tensor,
        matrix_low_1: torch.tensor,
        matrix_high_0: torch.tensor,
        matrix_high_1: torch.tensor
    ):
        """
        Args: DWT forward function to calculate four frequency-domain components: LL, LH, HL, HH
            ctx (function):
            x (torch.tensor): image input with shape [b, c, h, w]
            matrix_low_0 (torch.tensor): low frequency matrix
            matrix_low_1 (torch.tensor): low frequency matrix
            matrix_high_0 (torch.tensor): high frequency matrix
            matrix_high_1 (torch.tensor): high frequency matrix

        Returns:
            LL (torch.tensor): low-low frequency-domain components
            LH (torch.tensor): low-high frequency-domain components
            HL (torch.tensor): high-low frequency-domain components
            HH (torch.tensor): high-high frequency-domain components
        """
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
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_low_1.T), torch.matmul(grad_LH, matrix_high_1.T))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_low_1.T), torch.matmul(grad_HH, matrix_high_1.T))
        grad_input = torch.add(torch.matmul(matrix_low_0.T, grad_L), torch.matmul(matrix_high_0.T, grad_H))
        return grad_input, None, None, None, None


class IDWTFunction(Function):
    @staticmethod
    def forward(
        ctx,
        LL: torch.tensor,
        LH: torch.tensor,
        HL: torch.tensor,
        HH: torch.tensor,
        matrix_low_0: torch.tensor,
        matrix_low_1: torch.tensor,
        matrix_high_0: torch.tensor,
        matrix_high_1: torch.tensor
    ) -> torch.tensor:
        ctx.save_for_backward(matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)
        L = torch.add(torch.matmul(LL, matrix_low_1.T), torch.matmul(LH, matrix_high_1.T))
        H = torch.add(torch.matmul(HL, matrix_low_1.T), torch.matmul(HH, matrix_high_1.T))
        output = torch.add(torch.matmul(matrix_low_0.T, L), torch.matmul(matrix_high_0.T, H))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.tensor) -> torch.tensor:
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_low_0, grad_output)
        grad_H = torch.matmul(matrix_high_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_low_0)
        grad_LH = torch.matmul(grad_L, matrix_high_1)
        grad_HL = torch.matmul(grad_H, matrix_low_1)
        grad_HH = torch.matmul(grad_H, matrix_high_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        wavelet = pywt.Wavelet("haar")
        self.band_low = wavelet.rec_lo[::-1]
        self.band_high = wavelet.rec_hi[::-1]
        self.band_length = len(self.band_low)

    def generate_mat(self, height: int, width: int, device: str, dtype):
        max_side = max(height, width)
        half_side = max_side // 2
        mat_height = np.zeros((half_side, max_side + self.band_length - 2))
        mat_width = np.zeros((max_side - half_side, max_side + self.band_length - 2))
        end = None if self.band_length // 2 == 1 else -self.band_length // 2 + 1

        idx = 0
        for i in range(half_side):
            for j in range(self.band_length):
                mat_height[i, idx + j] = self.band_low[j]
            idx += self.band_length
        mat_height_0 = mat_height[0:height // 2, 0:height + self.band_length - 2]
        mat_height_1 = mat_height[0:width // 2, 0:width + self.band_length - 2]

        idx = 0
        for i in range(max_side - half_side):
            for j in range(self.band_length):
                mat_width[i, idx + j] = self.band_high[j]
            idx += self.band_length
        mat_width_0 = mat_width[0:height // 2, 0:height + self.band_length - 2]
        mat_width_1 = mat_width[0:width // 2, 0:width + self.band_length - 2]

        mat_height_0 = mat_height_0[:, self.band_length // 2 - 1:end]
        mat_height_1 = mat_height_1[:, self.band_length // 2 - 1:end]
        mat_height_1 = np.transpose(mat_height_1)

        mat_width_0 = mat_width_0[:, self.band_length // 2 - 1:end]
        mat_width_1 = mat_width_1[:, self.band_length // 2 - 1:end]
        mat_width_1 = np.transpose(mat_width_1)

        self.matrix_low_0 = torch.tensor(mat_height_0, device=device, dtype=dtype)
        self.matrix_low_1 = torch.tensor(mat_height_1, device=device, dtype=dtype)
        self.matrix_high_0 = torch.tensor(mat_width_0, device=device, dtype=dtype)
        self.matrix_high_1 = torch.tensor(mat_width_1, device=device, dtype=dtype)

    def forward(self, LL: torch.tensor, LH: torch.tensor, HL: torch.tensor, HH: torch.tensor) -> torch.tensor:
        device = LL.device
        batch, channels, height, width = LL.shape
        self.generate_mat(height * 2, width * 2, device, LL.dtype)
        return IDWTFunction.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0,
                                  self.matrix_high_1)


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
    def __init__(self, ):
        super().__init__()
        self.straight = FIMBlock()
        self.reverse = ReverseFIMBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.straight(x) + self.reverse(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels: int = None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels if out_channels is not None else in_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels if out_channels else in_channels // 2,
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


class Embedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.embd = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embd(x)
        return x


class CoarseSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
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

        x = x.view(batch, channels // self.head_dim, self.head_dim, -1).transpose(-2, -1)

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

        # [batch, num_heads, patch], [batch, num_heads, patch]
        output = torch.matmul(attn, v)
        output = output.permute(0, 1, 3, 2).reshape(batch, channels, h, w)
        output = self.upsample(output)
        return top_k, output


class RegionSelectionAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        self.course_attn = CoarseSelfAttention(dim, head_dim)
        self.topk_attn = TopkSelfAttention(head_dim)
        self.dwconv = DWConv(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_k, coarse_output = self.course_attn(x)
        region_output = self.topk_attn(coarse_output, top_k)
        x = coarse_output + region_output
        x = self.dwconv(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, dropout=0.):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else 4 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RSABlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.rsa = RegionSelectionAttention(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = MLP(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, dim, h, w = x.shape
        norm_x = x.flatten(2).permute(0, 2, 1)
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.permute(0, 1, 2).view(batch, dim, h, w)
        x = x + self.rsa(norm_x)
        norm_x = x.flatten(2).permute(0, 2, 1)
        norm_x = self.norm2(norm_x)
        norm_x = self.mlp(norm_x)
        norm_x = norm_x.permute(0, 1, 2).view(batch, dim, h, w)

        x = x + norm_x
        return x


class TopkSelfAttention(nn.Module):
    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.stride = 2
        self.kernel_size = 2
        self.head_dim = head_dim

        # self.embed = Embedding(dim, dim)

        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        residual = x.clone()

        # [batch, embeddings, height, width]
        batch, embed, h, w = x.shape
        # [batch, num_heads, embeddings, height, width]
        residual = residual.view(batch, embed // self.head_dim, self.head_dim, h, w)

        x = x.view(batch, embed // self.head_dim, self.head_dim, h, w)
        # [batch, num_heads, embeddings, height // 2, width // 2, 2, 2]
        patches = x.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
        patches = patches.reshape(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        # TODO remove loop with vectorize slicing
        batches = []
        for bat in range(batch):
            slices = []
            for i, inds in enumerate(top_k[bat]):
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
        _, num_heads, head_dim, patch, _, _, _ = residual_patches.shape
        residual_patches = residual_patches.reshape(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        output = output.view(batch, embed // self.head_dim, self.head_dim, -1, 2, 2)

        # torch.Size([1, 1, 64, 12321, 2, 2]) torch.Size([1, 1, 64, 3080, 2, 2])
        # The size of tensor a (3080) must match the size of tensor b (64) at non-singleton dimension 1
        # torch.Size([64, 3080, 2, 2]) torch.Size([64, 2, 2])
        for bat in range(batch):
            for i, inds in enumerate(top_k[bat]):
                for j, idx in enumerate(inds):
                    residual_patches[bat, i, :, idx, :, :] += output[bat, i, :, j, :, :]

        residual_patches = residual_patches.reshape(
            batch,
            -1,
            patch * 2,
            patch * 2
        )
        # residual_patches = residual_patches.view(batch, embed, h, w)
        return residual_patches


class DWTResidual(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
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
        self.upsample = Upsample(in_channels // 2, in_channels // 2)
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
        inverse_feature, upsample_feature = x.chunk(2, dim=-3)
        inverse = self.idwt(ll, lh, hl, inverse_feature)
        x = self.upsample(upsample_feature)
        x = torch.cat([inverse, x], dim=-3)
        x = self.conv(x)
        return x


class LAAModel(nn.Module):
    def __init__(self, in_channels=3, depths=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=depths[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.embed = Embedding(64, 64, 2)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.down_rsa_blocks = []
        self.downsample = []
        self.dwt = []
        for i, depth in enumerate(depths):
            self.down_rsa_blocks.append(
                RSABlock(depth)
            )
            self.dwt.append(
                DWTResidual(depth) if i != len(depths) - 1 else None
            )
            self.downsample.append(
                Downsample(depth) if i != len(depths) - 1 else None
            )

        # self.down_rsa_blocks = nn.ModuleList(self.down_rsa_blocks)
        # self.downsample = nn.ModuleList(self.downsample)
        # self.dwt = nn.ModuleList(self.dwt)

        self.feature_transformation = RSABlock(depths[-1])

        self.up_rsa_blocks = []
        self.wfm = []
        for i, depth in enumerate(depths[::-1]):
            depth *= 2
            self.up_rsa_blocks.append(
                RSABlock(depth)
            )
            self.wfm.append(
                WFM(depth) if i < len(depths) - 1 else None
            )
        # self.up_rsa_blocks = nn.ModuleList(self.up_rsa_blocks)
        # self.wfm = nn.ModuleList(self.wfm)

        self.conv2 = nn.Conv2d(
            in_channels=depths[0],
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(depths[-1] * 2, depths[-1] * 2, 1, 1, 0)
        self.act1x1 = nn.LeakyReLU()
        self.upsample = Upsample(depths[0] * 2, depths[0])

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x.clone()

        x = self.conv1(x)
        x = self.act1(x)

        x = self.embed(x)

        rsa_residual = []
        for rsa, downsample, dwt in zip(self.down_rsa_blocks, self.downsample, self.dwt):
            x = rsa(x)
            if dwt:
                rsa_residual.append(dwt(x))
            if downsample:
                x = downsample(x)

        rsa_residual.reverse()
        rsa_residual = rsa_residual + [None]
        x = torch.cat([self.feature_transformation(x), x], dim=-3)
        x = self.act1x1(self.conv1x1(x))
        for i, (rsa, wfm, wfm_features) in enumerate(zip(self.up_rsa_blocks, self.wfm, rsa_residual)):
            x = rsa(x)
            if wfm_features is not None and wfm is not None:
                x = wfm(*wfm_features, x)

        x = self.upsample(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = residual + x
        return x


if __name__ == "__main__":
    model = LAAModel()
    tensor = torch.rand(1, 3, 512, 512)
    assert model(tensor).shape == tensor.shape
    
    
