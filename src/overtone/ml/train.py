"""Offline training: fits the genre embedding (embedding.py), builds the
song-similarity NearestNeighbors index over it for the full track catalog, and
persists both for the live app to load.

Run: python -m overtone.ml.train
"""

import os

import joblib
from sklearn.neighbors import NearestNeighbors

from . import embedding, features
from .paths import (
    ARTIFACT_DIR,
    CLASSIFIER_FILE,
    CSV_PATH,
    EMBEDDING_FILE,
    FEATURES_FILE,
    INDEX_FILE,
    SCALER_FILE,
    TRACK_META_FILE,
    VARIANT_FILE,
)

N_NEIGHBORS = 4  # 3 similar tracks + the query track itself, which is always its own nearest neighbor
# (SimilarityModel requests extra at query time to absorb its duplicate filter)


def train(variant=features.DEFAULT_VARIANT, learned=True):
    catalog = features.load_catalog(CSV_PATH)
    X, scaler = features.build_feature_matrix(catalog, variant=variant)
    if learned:
        classifier, Z = embedding.fit(X, catalog[embedding.GENRE_LABEL].to_numpy())
    else:
        classifier, Z = None, X

    index = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="cosine")
    index.fit(Z)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, SCALER_FILE))
    joblib.dump(index, os.path.join(ARTIFACT_DIR, INDEX_FILE))
    # Both spaces are kept: the index is queried in the embedding, while the
    # lookup's duplicate check measures audio distance in the plain features
    joblib.dump(X, os.path.join(ARTIFACT_DIR, FEATURES_FILE))
    joblib.dump(Z, os.path.join(ARTIFACT_DIR, EMBEDDING_FILE))
    joblib.dump(variant, os.path.join(ARTIFACT_DIR, VARIANT_FILE))
    joblib.dump(classifier, os.path.join(ARTIFACT_DIR, CLASSIFIER_FILE))
    catalog[["track_id", "track_name", "track_artist", "alias_track_ids"]].to_csv(
        os.path.join(ARTIFACT_DIR, TRACK_META_FILE), index=False
    )

    stats = catalog.attrs["dedup"]
    print(
        f"Deduplicated {stats['track_ids']} track ids into {stats['songs']} songs "
        f"(merges: {stats['normalized_title']} by normalized title, "
        f"{stats['fuzzy_title_and_audio']} by fuzzy title + audio)"
    )
    print(f"Trained on {len(catalog)} unique songs using variant '{variant}'{' + genre embedding' if learned else ''}")
    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    train()
