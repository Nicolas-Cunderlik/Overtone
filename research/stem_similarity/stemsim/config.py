"""Locations, names and settings shared across the pipeline and the rating app."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # research/stem_similarity
# Gitignored. Raw downloads, extracted audio, stems and embeddings all live here.
DATA_DIR = Path(os.environ.get("STEMSIM_DATA_DIR", ROOT / "data"))
DB_FILE = "registry.sqlite"

# Identifies the client to the APIs we call (Last.fm asks for an identifiable agent)
USER_AGENT = "Overtone-research/0.1 (personal music-similarity research)"
HTTP_TIMEOUT_S = 30

# Stage names in the registry's stage_status. Kept here, not in the stage
# modules, so reading results doesn't import torch.
AUDIO_STAGE = "audio"
SEPARATE_STAGE = "separate"
EMBED_MODEL = "muq-mulan-large"
EMBED_STAGE = f"embed:{EMBED_MODEL}"

# Listener ratings. Committed (not under DATA_DIR): unlike everything else they
# can't be regenerated.
RATINGS_FILE = ROOT / "ratings.jsonl"
SESSIONS_DIR = DATA_DIR / "sessions"
PILOT = "pilot"  # session, design and audio source of the first app's ratings
DEFAULT_RATER = "nicolas"
RATING_PORT = 8765
