import pywt
import numpy as np
import torch
from torch.autograd import Function


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_low = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        self.band_high = [-1 / math.sqrt(2), 1 / math.sqrt(2)]
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        device = x.device
        batch, channels, height, width = x.shape
        self.generate_mat(height, width, device)
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
