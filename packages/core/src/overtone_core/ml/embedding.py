"""Learned genre embedding layered on top of the standardized audio features.

A gradient-boosted classifier learns to predict a song's genre from its audio
features, and its predicted genre probabilities are appended to the feature
vector as extra dimensions. The nearest-neighbor search then runs over "how it
sounds" plus "which genre it sounds like", so nonlinear genre cues the raw
distance can't express (e.g. high speechiness + mid tempo -> rap) pull
neighbors together.

Probabilities for the classifier's own training rows are out-of-fold: in-sample
predictions are overconfident, which would make catalog songs cluster tighter
by genre than a fresh song's prediction ever would.

GENRE_WEIGHT trades genre coherence against audio closeness - the probability
columns are the classifier's opinion, and weighting them up eventually picks
neighbors by predicted genre rather than by sound. Swept over 5 validation
splits of the training data (genre / subgenre precision@3, mean standardized
audio distance to the neighbors vs. no embedding):
    0: 0.396 / 0.164, +0.0%     2: 0.428 / 0.180, +1.2%     4: 0.434 / 0.182, +4.7%
    1: 0.415 / 0.174, +0.0%     3: 0.432 / 0.181, +2.9%     8: 0.436 / 0.181, +13.1%
Subgenre precision (which the classifier never sees) peaks around 4, but 2 gets
most of the gain while barely moving neighbors in audio space.
NeighborhoodComponentsAnalysis was also tried as a learned linear metric - it
never beat the hand-tuned weights (O(n^2) fitting limits it to a few thousand
songs, and the playlist genre labels are noisy).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

GENRE_LABEL = "playlist_genre"
GENRE_WEIGHT = 2.0
CV_FOLDS = 5


def fit(X, labels, weight=GENRE_WEIGHT, random_state=0):
    """Fit the genre classifier on `X` and return (classifier, embedding of X)."""
    classifier = HistGradientBoostingClassifier(random_state=random_state)
    proba = cross_val_predict(classifier, X, labels, cv=CV_FOLDS, method="predict_proba")
    classifier.fit(X, labels)
    return classifier, _combine(X, proba, weight)


def transform(X, classifier, weight=GENRE_WEIGHT):
    """Embed songs the classifier wasn't fit on, e.g. a held-out test split."""
    return _combine(X, classifier.predict_proba(X), weight)


def _combine(X, proba, weight):
    return np.hstack([X, weight * proba])
