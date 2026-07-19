import torch
from torch import nn

from src.model.mfm import LCNNBlock, MFMConv2d


class LCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            MFMConv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFMConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                in_channels=32,
                out_channels=48,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            MFMConv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=1,
            ),
            nn.BatchNorm2d(48),
            MFMConv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFMConv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            MFMConv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
