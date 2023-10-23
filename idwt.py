import pywt
import numpy as np
import torch
from torch.autograd import Function


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
        self.band_low = wavelet.rec_lo.reverse()
        self.band_high = wavelet.rec_hi.reverse()
        self.band_length = len(self.band_low)

    def generate_mat(self, height: int, width: int, device: str):
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

        self.matrix_low_0 = torch.tensor(mat_height_0, device=device)
        self.matrix_low_1 = torch.tensor(mat_height_1, device=device)
        self.matrix_high_0 = torch.tensor(mat_width_0, device=device)
        self.matrix_high_1 = torch.tensor(mat_width_1, device=device)

    def forward(self, LL: torch.tensor, LH: torch.tensor, HL: torch.tensor, HH: torch.tensor) -> torch.tensor:
        device = LL.device
        batch, channels, height, width = LL.shape
        self.generate_mat(height * 2, width * 2, device)
        return IDWTFunction.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)
