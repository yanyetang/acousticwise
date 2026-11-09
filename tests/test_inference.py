import numpy as np
import pytest

from acousticwise.inference import Predictor
from acousticwise.train import train_baseline, train_cnn
from tests.conftest import FREQS, tone


def test_predictor_loads_a_baseline_run_and_classifies_a_clip(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")

    prediction = Predictor.load(run_dir).predict(wav)

    assert prediction.label == "tug"
    assert list(prediction.scores) == ["cargo", "tug"]  # every class, in labels.json order


@pytest.mark.parametrize("model", ["baseline", "mean", "lstm"])  # baseline, CNN, CRNN
def test_predictor_scores_are_probabilities_for_every_model_kind(
    baseline_inputs, tmp_path, write_wav, model
):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    if model == "baseline":
        train_baseline(cfg, fold=1, run_dir=run_dir)
    else:
        cfg["cnn"].update(time_pooling=model, epochs=8, batch_size=4)
        train_cnn(cfg, fold=1, run_dir=run_dir)
    wav = write_wav(tone(5.0, 32000, freq=FREQS["cargo"], amplitude=0.2), 32000, name="upload.wav")

    prediction = Predictor.load(run_dir).predict(wav)

    assert prediction.label == "cargo"
    assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= s <= 1.0 for s in prediction.scores.values())


def test_a_long_file_scores_the_mean_of_its_windows(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    predictor = Predictor.load(run_dir)
    sr = 32000
    # 10 s: 5 s of cargo then 5 s of tug. Windows of 5 s every 2.5 s start at 0, 2.5 and 5 s.
    samples = np.concatenate(
        [
            tone(5.0, sr, freq=FREQS["cargo"], amplitude=0.2),
            tone(5.0, sr, freq=FREQS["tug"], amplitude=0.2),
        ]
    )
    long_wav = write_wav(samples, sr, name="long.wav")
    window_wavs = [
        write_wav(samples[int(start * sr) : int((start + 5.0) * sr)], sr, name=f"w{start}.wav")
        for start in (0.0, 2.5, 5.0)
    ]

    long_scores = predictor.predict(long_wav).scores

    window_scores = [predictor.predict(w).scores for w in window_wavs]
    for label in long_scores:
        expected = np.mean([s[label] for s in window_scores])
        assert long_scores[label] == pytest.approx(expected, abs=1e-6)
