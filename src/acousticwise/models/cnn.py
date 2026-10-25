"""The CNN and the CRNN: shared convolution layers, then a time-combining step.

Technical design, section 5.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from acousticwise.audio.cache import IndexEntry

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


def train_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    """One pass over the clips in a shuffled order; return the mean loss per clip.

    The clips can stay on the cpu: each batch is moved to the model's device.
    """
    model.train()
    device = next(model.parameters()).device
    order = torch.randperm(len(x), generator=generator)  # a new shuffle every epoch
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]  # the last batch may be shorter
        loss = F.cross_entropy(model(x[batch].to(device)), y[batch].to(device))
        optimizer.zero_grad()  # gradients add up unless cleared
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(order)


@torch.no_grad()  # no gradients needed to predict: faster, less memory
def predict_classes(model: nn.Module, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """The index of the highest-scoring class for each clip, on the cpu."""
    model.eval()  # dropout off, batch norm uses its saved averages
    device = next(model.parameters()).device
    batches = [
        model(x[i : i + batch_size].to(device)).argmax(dim=1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(batches)


def split_logmel(
    logmel: np.ndarray, index: dict[str, IndexEntry], entries: list[dict]
) -> tuple[np.ndarray, list[str]]:
    """Log-mels (clips, 64, 313) as float32, each shifted by its normalization offset, and
    labels, for one split's manifest entries in their order."""
    rows, labels, offsets = zip(*(index[e["id"]] for e in entries), strict=True)
    x = logmel[list(rows)].astype(np.float32)
    x += np.array(offsets, dtype=np.float32)[:, None, None]
    return x, list(labels)


def standardize(x: np.ndarray, scaling: dict[str, float]) -> np.ndarray:
    """Shift and scale log-mels by the training clips' mean and std, in place (no second copy).

    Training, evaluation and inference all call this, so inputs are always prepared alike.
    """
    x -= scaling["mean"]
    x /= scaling["std"]
    return x


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """The fastest device available: an NVIDIA GPU (cuda), then Apple silicon (mps), then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
