"""The CNN and the CRNN: shared convolution layers, then a time-combining step.

Technical design, section 5.
"""

import torch
from torch import nn

TIME_POOLING = ("mean", "lstm")


def _block(c_in: int, c_out: int, pool: tuple[int, int]) -> nn.Sequential:
    """A 3x3 convolution, batch norm, ReLU, then max pooling."""
    return nn.Sequential(
        # No bias: batch norm right after adds its own shift.
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(pool),
    )


class ConvNet(nn.Module):
    """Log-mel (batch, 64 mel bands, 313 frames) in, one score per class out.

    time_pooling="mean" averages over time (the CNN); "lstm" reads the time steps in order
    with a bidirectional LSTM (the CRNN).
    """

    def __init__(self, n_classes: int, time_pooling: str):
        super().__init__()
        if time_pooling not in TIME_POOLING:
            raise ValueError(f"Unknown time_pooling {time_pooling!r}. Use 'mean' or 'lstm'.")
        self.time_pooling = time_pooling
        # The first two blocks halve frequency and time; the last two only frequency,
        # so 78 time steps are left for the time-combining step.
        self.conv = nn.Sequential(
            _block(1, 32, (2, 2)),
            _block(32, 64, (2, 2)),
            _block(64, 128, (2, 1)),
            _block(128, 128, (2, 1)),
        )
        if time_pooling == "lstm":
            # 64 per direction; both directions' final states together give 128 values.
            self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))  # (batch, 128 channels, 4 mel rows, 78 time steps)
        h = h.mean(dim=2).transpose(1, 2)  # (batch, 78 time steps, 128 features)
        if self.time_pooling == "mean":
            z = h.mean(dim=1)
        else:
            _, (h_n, _) = self.lstm(h)  # h_n: (2 directions, batch, 64)
            z = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(self.dropout(z))  # (batch, n_classes)
