"""Offline training: fits the genre embedding (embedding.py), builds the
song-similarity NearestNeighbors index over it for the full track catalog, and
writes both as the next feature-knn version (models/feature-knn/vN/). The app
loads the version pinned in paths.py, so move the pin to use a new build.

Run: python -m overtone.ml.train
"""

import os

import joblib
from sklearn.neighbors import NearestNeighbors

from . import embedding, features, model_store
from .paths import (
    CLASSIFIER_FILE,
    CSV_PATH,
    EMBEDDING_FILE,
    FEATURE_MODEL,
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

    stats = catalog.attrs["dedup"]
    source = {"dataset": os.path.basename(CSV_PATH), "songs": len(catalog), "dedup": stats}
    params = {"variant": variant, "learned": learned, "n_neighbors": N_NEIGHBORS}
    with model_store.new_version(FEATURE_MODEL, source=source, params=params) as build:
        joblib.dump(scaler, build.path / SCALER_FILE)
        joblib.dump(index, build.path / INDEX_FILE)
        # Both spaces are kept: the index is queried in the embedding, while the
        # lookup's duplicate check measures audio distance in the plain features
        joblib.dump(X, build.path / FEATURES_FILE)
        joblib.dump(Z, build.path / EMBEDDING_FILE)
        joblib.dump(variant, build.path / VARIANT_FILE)
        joblib.dump(classifier, build.path / CLASSIFIER_FILE)
        catalog[["track_id", "track_name", "track_artist", "alias_track_ids"]].to_csv(
            build.path / TRACK_META_FILE, index=False
        )

    print(
        f"Deduplicated {stats['track_ids']} track ids into {stats['songs']} songs "
        f"(merges: {stats['normalized_title']} by normalized title, "
        f"{stats['fuzzy_title_and_audio']} by fuzzy title + audio)"
    )
    print(f"Trained on {len(catalog)} unique songs using variant '{variant}'{' + genre embedding' if learned else ''}")
    print(f"Wrote {FEATURE_MODEL} {build.version} to {build.path} (the app loads the version pinned in paths.py)")


if __name__ == "__main__":
    train()
