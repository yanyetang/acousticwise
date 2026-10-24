import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from acousticwise.audio.features import mfcc_summary
from acousticwise.models.baseline import make_baseline, mfcc_features


def test_mfcc_features_gives_one_row_per_clip_after_adding_its_offset():
    rng = np.random.default_rng(0)
    # float16, like the cache; the offsets shift each clip's log-mel by a constant dB.
    logmel = rng.uniform(-80, -20, size=(3, 64, 313)).astype(np.float16)
    offsets_db = np.array([0.0, 10.0, -5.0])

    features = mfcc_features(logmel, offsets_db, n_mfcc=20)

    assert features.shape == (3, 40)
    assert features.dtype == np.float32
    expected = mfcc_summary(logmel[1].astype(np.float32) + 10.0, n_mfcc=20)
    np.testing.assert_allclose(features[1], expected, rtol=1e-5)


@pytest.mark.parametrize("classifier", ["svm", "random_forest"])
def test_baseline_scales_then_learns_two_separate_classes(classifier):
    rng = np.random.default_rng(0)
    # Two clouds of 40-number feature rows, far apart, on a large scale like MFCC means.
    x = np.vstack([rng.normal(-100, 5, size=(30, 40)), rng.normal(100, 5, size=(30, 40))])
    y = ["cargo"] * 30 + ["tug"] * 30

    model = make_baseline(classifier, seed=0).fit(x, y)

    assert isinstance(model.steps[0][1], StandardScaler)
    assert list(model.predict([[-100.0] * 40, [100.0] * 40])) == ["cargo", "tug"]
    # The app shows a score per class, so the model must give probabilities.
    assert model.predict_proba([[100.0] * 40]).shape == (1, 2)


def test_baseline_refuses_an_unknown_classifier():
    with pytest.raises(ValueError, match="knn"):
        make_baseline("knn", seed=0)
