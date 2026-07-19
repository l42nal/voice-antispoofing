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
            nn.MaxPool2d(2, 2),
            LCNNBlock(
                in_channels=32,
                hidden_channels=32,
                out_channels=48,
            ),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2, 2),
            LCNNBlock(
                in_channels=48,
                hidden_channels=48,
                out_channels=64,
            ),
            nn.MaxPool2d(2, 2),
            LCNNBlock(
                in_channels=64,
                hidden_channels=64,
                out_channels=32,
            ),
            nn.BatchNorm2d(32),
            LCNNBlock(
                in_channels=32,
                hidden_channels=32,
                out_channels=32,
            ),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.features(x)
