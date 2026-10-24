import numpy as np

from acousticwise.audio.features import mfcc_summary
from acousticwise.models.baseline import mfcc_features


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
