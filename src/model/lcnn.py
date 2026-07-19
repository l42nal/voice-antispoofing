import torch
from torch import nn

from src.model.mfm import LCNNBlock, MFMConv2d, MFMLinear


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


class LCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.75,
    ):
        super().__init__()

        self.backbone = LCNNBackbone()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            MFMLinear(
                in_features=18944,
                out_features=80,
            ),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(80),
            nn.Linear(
                in_features=80,
                out_features=num_classes,
            ),
        )

    def forward(
        self,
        audio: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        logits = self.backbone(audio)
        logits = self.classifier(logits)

        return {"logits": logits}
