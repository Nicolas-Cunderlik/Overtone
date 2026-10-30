"""Locations and settings shared by the app and the research pipeline.

Paths never depend on the working directory. PROJECT_ROOT holds models/
and the .env with API keys (unless OVERTONE_ENV_FILE points elsewhere):
  - in a source checkout (this package installed editable from
    packages/core), the repo root, which also holds data/ and research/;
  - in an install made by the installer, the install folder, the parent of
    the app's venv (%LOCALAPPDATA%\\Programs\\Overtone).
App-only settings are in overtone.config, research-only ones in
stemsim.config.
"""

import os
import sys
from pathlib import Path


def find_root(module_file=__file__, prefix=sys.prefix):
    """(root, installed): the repo root when `module_file` lies in a source
    checkout, else the folder holding the venv at `prefix`."""
    parents = Path(module_file).resolve().parents
    checkout = parents[4] if len(parents) > 4 else None
    if checkout and (checkout / "packages" / "core" / "pyproject.toml").is_file():
        return str(checkout), False
    return str(Path(prefix).resolve().parent), True


PROJECT_ROOT, INSTALLED = find_root()

ENV_FILE = os.environ.get("OVERTONE_ENV_FILE", os.path.join(PROJECT_ROOT, ".env"))

# Downloaded model weights (Demucs, MuQ-MuLan). An install keeps its own, so
# uninstalling frees them; a checkout uses the default torch/Hugging Face caches.
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights") if INSTALLED else None

# Identifies the client to the APIs we call (Last.fm asks for an identifiable agent)
USER_AGENT = "Overtone-research/0.1 (personal music-similarity research)"
HTTP_TIMEOUT_S = 30
