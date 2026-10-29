"""Locations and settings shared by the app and the research pipeline.

Paths never depend on the working directory. PROJECT_ROOT is the repo root
(this package is installed editable from packages/core); models/ and data/
live there, and so does the .env with API keys unless OVERTONE_ENV_FILE
points elsewhere. App-only settings are in overtone.config, research-only
ones in stemsim.config.
"""

import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])

ENV_FILE = os.environ.get("OVERTONE_ENV_FILE", os.path.join(PROJECT_ROOT, ".env"))

# Identifies the client to the APIs we call (Last.fm asks for an identifiable agent)
USER_AGENT = "Overtone-research/0.1 (personal music-similarity research)"
HTTP_TIMEOUT_S = 30
