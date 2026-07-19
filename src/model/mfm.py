import torch
from torch import nn


class MFM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError("Number of channels must be even, " f"got {x.shape[1]}")

        x = x.reshape(
            x.shape[0],
            x.shape[1] // 2,
            2,
            x.shape[2],
            x.shape[3],
        )

        x, _ = x.max(dim=2)

        return x


class MFMConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.mfm = MFM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.mfm(x)

        return x


class LCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = MFMConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.conv2 = MFMConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x
