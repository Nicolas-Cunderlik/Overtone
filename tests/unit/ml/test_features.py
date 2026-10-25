import numpy as np
import pandas as pd
import pytest

from overtone.ml import features


def key_points(keys):
    encoded = features.add_derived_columns(pd.DataFrame({"key": keys}))
    return encoded[["key_sin", "key_cos"]].to_numpy()


def test_key_encoding_is_circular():
    # B (11) is a semitone from C (0), exactly like C to C# (1)
    b, c, c_sharp = key_points([11, 0, 1])
    assert np.linalg.norm(b - c) == pytest.approx(np.linalg.norm(c - c_sharp))


def test_undetected_key_sits_equally_far_from_every_key():
    # Spotify's legacy format reports key -1 when no key was detected
    undetected, *keys = key_points([-1, *range(12)])
    np.testing.assert_allclose(undetected, [0.0, 0.0])
    distances = [np.linalg.norm(undetected - key) for key in keys]
    assert distances == pytest.approx([1.0] * 12)
