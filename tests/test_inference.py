from acousticwise.inference import Predictor
from acousticwise.train import train_baseline
from tests.conftest import FREQS, tone


def test_predictor_loads_a_baseline_run_and_classifies_a_clip(baseline_inputs, tmp_path, write_wav):
    cfg, _ = baseline_inputs
    run_dir = tmp_path / "run"
    train_baseline(cfg, fold=1, run_dir=run_dir)
    wav = write_wav(tone(5.0, 32000, freq=FREQS["tug"], amplitude=0.2), 32000, name="upload.wav")

    prediction = Predictor.load(run_dir).predict(wav)

    assert prediction.label == "tug"
    assert list(prediction.scores) == ["cargo", "tug"]  # every class, in labels.json order
