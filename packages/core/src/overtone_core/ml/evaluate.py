"""Evaluation: holds out a stratified test split of songs, queries each held-out
song against an index built from the rest, and checks whether the neighbors
share its genre - a proxy for "are the neighbors actually similar", since the
dataset has no ground truth for similarity itself.

Metrics, each at k = 1, 3 (what the app shows) and 5:
- genre precision@k: fraction of the k neighbors with the query's genre
  (6 broad genres). This is the share of a recommendation list that is
  on-genre, i.e. what the user actually sees.
- subgenre precision@k: the same over the 24 playlist subgenres - a much
  stricter test (e.g. "neo soul" vs. "new jack swing", both r&b).
- genre vote accuracy@k: whether the majority genre of the k neighbors is the
  query's (kNN classification). Kept for comparison with earlier runs; it
  hides how many of the k were off-genre.
Each precision is printed next to its chance baseline - the precision a
random neighbor would score given the genre mix.

The catalog is deduplicated first (dedup.py), so the split is by song: copies
of one song can't land on both sides and be found as each other's neighbor,
which would inflate every metric. The leak check line verifies that.

Compares the named feature variants from features.py so feature selection is
driven by measured accuracy, not an assumed feature count, plus the default
variant with the learned genre embedding on top (embedding.py) - what train.py
actually ships. Its classifier is fit on the train split only. It's trained on
genre, so part of its genre-precision gain is by construction; subgenre
precision is the fairer read.

Run: python -m overtone_core.ml.evaluate
"""

from collections import Counter
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from . import dedup, embedding, features
from .paths import CSV_PATH

K_VALUES = (1, 3, 5)


def split(catalog, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(catalog, test_size=0.2, random_state=seed, stratify=catalog["playlist_genre"])
    return cast(pd.DataFrame, train_df), cast(pd.DataFrame, test_df)  # sklearn's stubs say arrays


def count_leaks(train_df, test_df):
    """Held-out songs that also appear (as a near-duplicate) in the train split."""
    train_keys = {dedup.song_key(t, a) for t, a in zip(train_df["track_name"], train_df["track_artist"], strict=True)}
    return sum(
        dedup.song_key(t, a) in train_keys for t, a in zip(test_df["track_name"], test_df["track_artist"], strict=True)
    )


def chance_precision(train_labels, test_labels):
    """Expected precision of a uniformly random neighbor."""
    p_train = Counter(train_labels)
    n_train = len(train_labels)
    return float(np.mean([p_train[label] / n_train for label in test_labels]))


def evaluate_variant(train_df, test_df, variant, learned=False, k_values=K_VALUES):
    X_train, scaler = features.build_feature_matrix(train_df, variant=variant)
    X_test, _ = features.build_feature_matrix(test_df, variant=variant, scaler=scaler)
    if learned:
        classifier, X_train = embedding.fit(X_train, train_df[embedding.GENRE_LABEL].to_numpy())
        X_test = embedding.transform(X_test, classifier)

    nn = NearestNeighbors(n_neighbors=max(k_values), metric="cosine")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)

    results = {}
    for label in ("playlist_genre", "playlist_subgenre"):
        train_labels = train_df[label].to_numpy()
        test_labels = test_df[label].to_numpy()
        neighbor_labels = train_labels[indices]  # (n_test, max_k)
        matches = neighbor_labels == test_labels[:, None]
        for k in k_values:
            results[(label, "precision", k)] = matches[:, :k].mean()
        if label == "playlist_genre":
            for k in k_values:
                votes = [Counter(row[:k]).most_common(1)[0][0] for row in neighbor_labels]
                results[(label, "vote", k)] = np.mean(np.array(votes) == test_labels)
    return results


def main():
    catalog = features.load_catalog(CSV_PATH)
    stats = catalog.attrs["dedup"]
    print(
        f"Catalog: {stats['track_ids']} track ids -> {stats['songs']} songs after dedup, "
        f"{catalog['playlist_genre'].nunique()} genres / "
        f"{catalog['playlist_subgenre'].nunique()} subgenres"
    )

    train_df, test_df = split(catalog)
    print(
        f"Split: {len(train_df)} train / {len(test_df)} test songs; "
        f"leak check - test songs with a copy in train: {count_leaks(train_df, test_df)}\n"
    )

    genre_chance = chance_precision(train_df["playlist_genre"], test_df["playlist_genre"])
    subgenre_chance = chance_precision(train_df["playlist_subgenre"], test_df["playlist_subgenre"])
    majority = test_df["playlist_genre"].value_counts(normalize=True).iloc[0]
    print(
        f"Chance baselines: genre precision {genre_chance:.3f}, subgenre precision "
        f"{subgenre_chance:.3f}, genre vote (always predict most common) {majority:.3f}\n"
    )

    header = f"{'variant':28s} {'metric':22s} " + " ".join(f"{'k=' + str(k):>6s}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    runs = [(variant, variant, False) for variant in features.FEATURE_VARIANTS]
    runs.append((f"{features.DEFAULT_VARIANT}+genre", features.DEFAULT_VARIANT, True))
    for run_name, variant, learned in runs:
        results = evaluate_variant(train_df, test_df, variant, learned=learned)
        rows = [
            ("genre precision@k", "playlist_genre", "precision"),
            ("subgenre precision@k", "playlist_subgenre", "precision"),
            ("genre vote accuracy@k", "playlist_genre", "vote"),
        ]
        for i, (name, label, kind) in enumerate(rows):
            scores = " ".join(f"{results[(label, kind, k)]:6.3f}" for k in K_VALUES)
            print(f"{run_name if i == 0 else '':28s} {name:22s} {scores}")


if __name__ == "__main__":
    main()
