import numpy as np
import pytest
from overtone_core.ml import dedup


@pytest.mark.parametrize(
    "title, expected",
    [
        # Packaging-only qualifiers are dropped...
        ("Wake Me Up - Radio Edit", "wake me up"),
        ("Bad Guy (feat. Justin Bieber)", "bad guy"),
        ("Bohemian Rhapsody - Remastered 2011", "bohemian rhapsody"),
        # ...but ones that mean a different recording are kept
        ("One Kiss (with Dua Lipa) - Oliver Heldens Remix", "one kiss oliver heldens remix"),
        ("Hotel California - Live", "hotel california live"),
    ],
)
def test_normalize_title(title, expected):
    assert dedup.normalize_title(title) == expected


def test_same_song_under_packaging_variants_is_duplicate():
    assert dedup.is_duplicate("Wake Me Up - Radio Edit", "Avicii", "Wake Me Up", "Avicii")


def test_same_title_by_different_artists_is_not_duplicate():
    assert not dedup.is_duplicate("Hello", "Adele", "Hello", "Lionel Richie")


def test_similar_titles_need_matching_audio_to_merge():
    # "Part 1" vs "Part 2" fuzzy-match as strings - only near-identical audio confirms a copy
    same_audio = np.zeros(3), np.zeros(3)
    different_audio = np.zeros(3), np.ones(3)
    assert dedup.is_duplicate("Song Part 1", "X", "Song Part 2", "X", *same_audio)
    assert not dedup.is_duplicate("Song Part 1", "X", "Song Part 2", "X", *different_audio)
    assert not dedup.is_duplicate("Song Part 1", "X", "Song Part 2", "X")
