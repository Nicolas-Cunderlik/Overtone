"""Locations and network etiquette shared by every stage."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # research/stem_similarity
# Gitignored. Raw downloads, extracted audio, stems and embeddings all live here.
DATA_DIR = Path(os.environ.get("STEMSIM_DATA_DIR", ROOT / "data"))
DB_FILE = "registry.sqlite"

# Identifies the client to the APIs we call (Last.fm asks for an identifiable agent)
USER_AGENT = "Overtone-research/0.1 (personal music-similarity research)"
HTTP_TIMEOUT_S = 30
