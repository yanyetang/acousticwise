import pytest
import torch

from acousticwise.models.cnn import ConvNet, train_epoch


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
