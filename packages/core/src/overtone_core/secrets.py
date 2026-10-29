"""API keys from the .env file (config.ENV_FILE).

Kept free of heavy imports so anything - the app, the research pipeline,
scripts - can read a key without pulling in spotipy or openai. The file is
read on first use rather than at import, so a missing file fails only the
client that needed it (the services catch and log that) instead of crashing
the app at startup. Values are never logged.
"""

import functools

from .config import ENV_FILE


@functools.lru_cache(maxsize=1)
def _load_env_file():
    values = {}
    with open(ENV_FILE) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def get_secret(name):
    """A key from the .env file, or None if it isn't set there."""
    return _load_env_file().get(name)
