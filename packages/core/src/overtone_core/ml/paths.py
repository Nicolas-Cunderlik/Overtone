"""Filesystem locations shared by the ML scripts and the live model.

Data and trained models live outside the package, at the project root:
data/ holds the downloaded dataset, models/<model>/<version>/ each trained
model (model_store.py). A builder writes a version and a loader reads it, so model
names, pinned versions and filenames live here as the contract between the two.
"""

import os

from ..config import PROJECT_ROOT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "spotify_songs.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Models, and the version of each that the app loads. A new build is a new
# version and changes nothing until its pin here is moved.
FEATURE_MODEL = "feature-knn"  # Spotify audio features + genre embedding, cosine kNN (train.py)
FEATURE_MODEL_VERSION = "v1"
STEM_MODEL = "stem-muq"  # MuQ-MuLan embeddings of the mix and Demucs stems, fitted channel weights
# (built by research/stem_similarity: python -m stemsim index; loaded by stem_model.py)
STEM_MODEL_VERSION = "v1"

# feature-knn files
INDEX_FILE = "index.joblib"  # NearestNeighbors fit on the embedding
FEATURES_FILE = "features.joblib"  # standardized audio features (dedup distance space)
EMBEDDING_FILE = "embedding.joblib"  # features + genre probabilities (search space)
# Everything needed to embed a song outside the catalog the same way
SCALER_FILE = "scaler.joblib"
VARIANT_FILE = "variant.joblib"
CLASSIFIER_FILE = "classifier.joblib"  # genre classifier; None without the embedding
TRACK_META_FILE = "track_meta.csv"

# stem-muq files; every array's rows follow STEM_TRACKS_FILE
STEM_TRACKS_FILE = "tracks.csv"  # song id, Spotify track id, title, artist
STEM_CHANNEL_FILE = "{channel}.npy"  # (songs, dims) float16 unit vectors, one file per channel in params["channels"]
STEM_LEVELS_FILE = "rms_db.npy"  # (songs, channels) float32 RMS level in dB, for silence gating
