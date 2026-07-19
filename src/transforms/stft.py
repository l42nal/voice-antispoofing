import torch
from torch import nn


class STFTTransform(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps

        self.register_buffer(
            "window",
            torch.hann_window(win_length),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(
                "STFTTransform expects audio with shape "
                f"[batch_size, time], but got {tuple(audio.shape)}"
            )

        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        magnitude = spectrum.abs()
        log_magnitude = torch.log(magnitude + self.eps)

        return log_magnitude.unsqueeze(1)
