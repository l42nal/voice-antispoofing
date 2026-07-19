import torch
import torch.nn.functional as F
from torch import nn


class STFTTransform(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        eps: float = 1e-6,
        max_frames: int = 600,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        self.max_frames = max_frames
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        spectrogram = stft.abs()
        spectrogram = torch.log(spectrogram + self.eps)
        spectrogram = spectrogram.unsqueeze(1)

        frames = spectrogram.shape[-1]

        if frames > self.max_frames:
            spectrogram = spectrogram[..., : self.max_frames]

        elif frames < self.max_frames:
            spectrogram = F.pad(
                spectrogram,
                pad=(0, self.max_frames - frames),
            )

        return spectrogram
