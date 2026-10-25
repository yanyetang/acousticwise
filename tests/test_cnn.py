import pytest
import torch

from acousticwise.models.cnn import ConvNet


@pytest.mark.parametrize("time_pooling", ["mean", "lstm"])  # CNN, CRNN
def test_forward_gives_one_score_per_class_for_each_clip(time_pooling):
    model = ConvNet(n_classes=12, time_pooling=time_pooling)
    batch = torch.randn(3, 64, 313)  # 3 clips of 64 mel bands x 313 frames

    logits = model(batch)

    assert logits.shape == (3, 12)
