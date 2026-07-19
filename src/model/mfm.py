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
