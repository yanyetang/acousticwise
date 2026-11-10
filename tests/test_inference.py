import json

import numpy as np
import pytest

from acousticwise.errors import (
    AudioTooShortError,
    FileTooLargeError,
    InferenceError,
    SilentAudioError,
    UnsupportedFileError,
)
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


def _not_a_wav(tmp_path, write_wav):
    path = tmp_path / "notes.wav"
    path.write_bytes(b"this is not audio")
    return path


# One case per row of the design's error table. The run is trained with small limits
# (1 MB, 12 s), so each case stays small: 32 kHz 32-bit audio is 128 kB per second.
ERROR_CASES = {
    "unreadable": (_not_a_wav, UnsupportedFileError),
    "too large": (  # 10 s at 32 kHz, 32-bit: 1.28 MB, under the duration limit
        lambda tmp_path, write_wav: write_wav(tone(10.0, 32000), 32000, name="big.wav"),
        FileTooLargeError,
    ),
    "too short": (
        lambda tmp_path, write_wav: write_wav(tone(3.9, 32000), 32000, name="short.wav"),
        AudioTooShortError,
    ),
    "silent": (
        lambda tmp_path, write_wav: write_wav(np.zeros(5 * 32000, "float32"), 32000),
        SilentAudioError,
    ),
}


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_each_invalid_upload_raises_its_typed_error(baseline_inputs, tmp_path, write_wav, case):
    cfg, _ = baseline_inputs
    cfg["validation"].update(max_file_mb=1, max_seconds=12)  # saved in the run's config.json
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    make_file, error = ERROR_CASES[case]

    with pytest.raises(error):
        Predictor.load(run_dir).predict(make_file(tmp_path, write_wav))


def test_a_file_exactly_at_the_size_limit_is_classified(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")
    # The limit is this file's own size; MB as Streamlit counts its upload cap.
    cfg["validation"]["max_file_mb"] = wav.stat().st_size / (1024 * 1024)
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)

    assert Predictor.load(run_dir).predict(wav).label == "tug"


def test_audio_exactly_at_the_duration_limit_is_classified(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    cfg["validation"]["max_seconds"] = 12
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    # 12 s at the run's own 32 kHz: 384000 samples, so no resampling rounds the length.
    wav = write_wav(tone(12.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")

    prediction = Predictor.load(run_dir).predict(wav)

    assert prediction.label == "tug"
    assert not prediction.trimmed


def test_audio_over_the_duration_limit_is_classified_on_its_first_max_seconds(
    baseline_inputs, tmp_path, write_wav
):
    cfg, _ = baseline_inputs
    cfg["validation"]["max_seconds"] = 12
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    # 12 s of tug, then 30 s of cargo: scored whole, the cargo windows would outvote the tug.
    tug = tone(12.0, 32000, freq=FREQS["tug"], amplitude=0.2)
    cargo = tone(30.0, 32000, freq=FREQS["cargo"], amplitude=0.2)
    wav = write_wav(np.concatenate([tug, cargo]), 32000, name="upload.wav")

    prediction = Predictor.load(run_dir).predict(wav)

    assert prediction.label == "tug"
    assert prediction.trimmed
    assert len(prediction.waveform) == 12 * 32000  # the plots show what was classified


def test_an_unexpected_model_error_becomes_an_inference_error(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    # A broken run: the model was trained on 20 MFCCs, the config now says 13, so the
    # model gets the wrong number of features and scikit-learn raises a ValueError.
    record = json.loads((run_dir / "config.json").read_text())
    record["config"]["features"]["n_mfcc"] = 13
    (run_dir / "config.json").write_text(json.dumps(record))
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")

    with pytest.raises(InferenceError) as excinfo:
        Predictor.load(run_dir).predict(wav)

    assert isinstance(excinfo.value.__cause__, ValueError)  # the original kept for debugging


def test_labels_that_do_not_match_the_model_raise_an_inference_error(
    baseline_inputs, tmp_path, write_wav
):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    # A broken run: labels.json lists a third class, but the model scores only two.
    (run_dir / "labels.json").write_text(json.dumps(["cargo", "tug", "pilotvessel"]))
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")

    with pytest.raises(InferenceError) as excinfo:
        Predictor.load(run_dir).predict(wav)

    assert isinstance(excinfo.value.__cause__, ValueError)  # not scores with a class missing
