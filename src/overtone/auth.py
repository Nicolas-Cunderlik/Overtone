"""
Filename: src/overtone/auth.py
Author: Nicolas Cunderlik
Date: 2025
Description: Authorization tools for accessing the Spotify and OpenAI services.

Copyright © 2025 Nicolas Cunderlik. All Rights Reserved.

This source code is the property of the author/owner and is protected under
copyright law. Unauthorized copying, modification, distribution, or any use
of this file, in part or in full, without prior written permission from the
author/owner, is strictly prohibited.

For inquiries, contact: nicolas7cunderlik@gmail.com
"""

import os

import spotipy
from openai import OpenAI
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

from .config import SPOTIFY_TOKEN_CACHE_PATH
from .secrets import get_secret


# Spotify authentication and client setup
def get_spotify_client():
    """
    Returns a Spotify client object for interacting with the Spotify API.

    The client will use the client ID, client secret, and redirect URI
    set as SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and
    SPOTIPY_REDIRECT_URI in the .env file, respectively.

    The client uses the "user-read-playback-state" scope to read the user's
    currently playing track, and "user-modify-playback-state" to play a
    clicked similar track (playback control requires Spotify Premium).
    spotipy re-prompts for login automatically if the cached token predates
    a scope added here.

    Returns:
        A Spotify client object.
    """
    scope = "user-read-playback-state user-modify-playback-state"
    os.makedirs(os.path.dirname(SPOTIFY_TOKEN_CACHE_PATH), exist_ok=True)
    sp_oauth = SpotifyOAuth(
        client_id=get_secret('SPOTIPY_CLIENT_ID'),
        client_secret=get_secret('SPOTIPY_CLIENT_SECRET'),
        redirect_uri=get_secret('SPOTIPY_REDIRECT_URI'),
        scope=scope,
        cache_handler=CacheFileHandler(cache_path=SPOTIFY_TOKEN_CACHE_PATH),
    )
    return spotipy.Spotify(auth_manager=sp_oauth)


def get_openai_client():
    """
    Returns an OpenAI client authenticated with OPENAI_API_KEY from the .env file.

    Returns:
        An OpenAI client object.
    """
    return OpenAI(api_key=get_secret('OPENAI_API_KEY'))
