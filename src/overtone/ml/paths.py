"""Filesystem locations shared by the ML scripts and the live model.

Data and trained models live outside the package, at the project root:
data/ holds the downloaded dataset, models/similarity/ the trained artifacts
(bundled into the exe under the same relative path). train.py writes the
artifacts and similarity_model.py reads them, so the filenames live here as
the single contract between the two.
"""

import os

from ..config import PROJECT_ROOT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "spotify_songs.csv")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "models", "similarity")

INDEX_FILE = "index.joblib"  # NearestNeighbors fit on the embedding
FEATURES_FILE = "features.joblib"  # standardized audio features (dedup distance space)
EMBEDDING_FILE = "embedding.joblib"  # features + genre probabilities (search space)
# Everything needed to embed a song outside the catalog the same way
SCALER_FILE = "scaler.joblib"
VARIANT_FILE = "variant.joblib"
CLASSIFIER_FILE = "classifier.joblib"  # genre classifier; None without the embedding
TRACK_META_FILE = "track_meta.csv"
