import pytest
import torch

from acousticwise.models.cnn import ConvNet, pick_device, set_seed, train_epoch


@pytest.mark.parametrize("time_pooling", ["mean", "lstm"])  # CNN, CRNN
def test_forward_gives_one_score_per_class_for_each_clip(time_pooling):
    model = ConvNet(n_classes=12, time_pooling=time_pooling)
    batch = torch.randn(3, 64, 313)  # 3 clips of 64 mel bands x 313 frames

    logits = model(batch)

    assert logits.shape == (3, 12)


def two_class_clips(n_per_class=8):
    """Noise log-mels; class 1 clips are louder in the lowest 16 mel bands, so it can be learned."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2 * n_per_class, 64, 313, generator=gen)
    x[n_per_class:, :16, :] += 2.0
    y = torch.tensor([0] * n_per_class + [1] * n_per_class)
    return x, y


@pytest.mark.parametrize("time_pooling", ["mean", "lstm"])
def test_a_few_epochs_lower_the_training_loss(time_pooling):
    torch.manual_seed(0)
    x, y = two_class_clips()
    model = ConvNet(n_classes=2, time_pooling=time_pooling)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(0)

    losses = [
        train_epoch(model, x, y, optimizer, batch_size=4, generator=shuffle) for _ in range(5)
    ]

    # The loss is the mean over the epoch's batches; learning shows as a lower loss.
    assert losses[-1] < losses[0]


def losses_for_seed(seed, time_pooling, epochs=2):
    """Seed everything, then train a fresh model; nothing else is seeded by hand."""
    set_seed(seed)
    x, y = two_class_clips()
    model = ConvNet(n_classes=2, time_pooling=time_pooling)  # random starting weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(seed)
    return [
        train_epoch(model, x, y, optimizer, batch_size=4, generator=shuffle) for _ in range(epochs)
    ]


@pytest.mark.parametrize("time_pooling", ["mean", "lstm"])
def test_the_same_seed_gives_the_same_losses(time_pooling):
    assert losses_for_seed(7, time_pooling) == losses_for_seed(7, time_pooling)
    # And the seed matters: another one starts from other weights.
    assert losses_for_seed(7, time_pooling) != losses_for_seed(8, time_pooling)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),  # Kaggle's GPU first
        (False, True, "mps"),  # then an Apple-silicon GPU
        (False, False, "cpu"),  # then the CPU, which always works
    ],
)
def test_pick_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    # Pretend which GPUs exist, so the test gives the same answer on any machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert pick_device() == torch.device(expected)
