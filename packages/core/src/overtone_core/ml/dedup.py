"""Near-duplicate detection for the track catalog.

The dataset lists the same song under several Spotify track IDs (single vs.
album release, remasters, "feat." credits, radio edits, compilations), which
lets the kNN model recommend a song to itself. Two cheap rules, both scoped to
tracks by the same artist, identify those copies:

1. Exact match on a normalized title - packaging-only qualifiers such as
   "(feat. X)", "- Remastered 2009" or "- Radio Edit" are stripped, while
   qualifiers that mean a genuinely different recording ("Remix", "Live",
   "Acoustic", ...) are kept as part of the title.
2. Fuzzy title match (stdlib difflib) confirmed by near-identical audio
   features - catches copies whose qualifier isn't in the keyword lists, e.g.
   "Sex" vs. "Sex (Cheat Codes X Kris Kross Amsterdam)".

Fuzzy title matching alone is deliberately not a rule: "Part 1" vs. "Part 2"
or "Song I" vs. "Song II" score as near-identical strings but are different
songs. Audio alone isn't one either - remasters and explicit/clean edits of
the same song often sit far apart in feature space, so it can confirm a
match but not rule one out.

No third-party fuzzy-matching dependency: comparisons are only made within an
artist (largest artist in the catalog has ~130 tracks), so difflib is cheap.
"""

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

# Qualifiers that only describe packaging/credits - same underlying recording
_PACKAGING = re.compile(
    r"\b(feat|ft|featuring|with|remaster(ed)?|radio (edit|mix|version)|"
    r"edit|single|album|original (mix|version)|version|mono|stereo|explicit|clean|"
    r"deluxe|bonus|from|soundtrack|anniversary|edition)\b|\b(19|20)\d{2}\b"
)
# Qualifiers that mean a different recording - kept as part of the title.
# Checked after removing "original mix"/"radio mix", which are canonical cuts.
_DISTINCT = re.compile(
    r"\b(remix|rmx|mix|live|en vivo|acoustic|unplugged|instrumental|extended|club|"
    r"vip|demo|karaoke|cover|sped up|slowed|reverb|piano|orchestral|acapella|"
    r"a cappella|dub|bootleg|rework|flip)\b"
)
_CANONICAL_MIX = re.compile(r"\b(original|radio) mix\b")
_FRAGMENT = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
_INLINE_FEAT = re.compile(r"\s(feat|ft|featuring)\b.*$")

# Fuzzy title similarity (0-1) needed for rule 2, and the max standardized
# feature distance treated as "the same audio". Measured on this catalog:
# same-artist pairs below 0.05 are almost all the same recording, whereas a
# random same-artist pair sits around 3.9.
FUZZY_TITLE_RATIO = 0.6
AUDIO_DUPLICATE_DISTANCE = 0.05


def _fold(text):
    """Lowercase and strip accents so 'Beyoncé' == 'Beyonce'."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _clean(text):
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keep_fragment(fragment):
    without_canonical = _CANONICAL_MIX.sub("", fragment)
    if _DISTINCT.search(without_canonical):
        return True
    # Unknown qualifiers are kept (conservative) - rule 2 can still merge them
    return not _PACKAGING.search(fragment)


def normalize_title(title):
    """Title with packaging-only qualifiers removed, e.g.
    'Wake Me Up - Radio Edit' -> 'wake me up',
    'One Kiss (with Dua Lipa) - Oliver Heldens Remix' -> 'one kiss oliver heldens remix'."""
    text = _fold(title)
    fragments = _FRAGMENT.findall(text)
    text = _FRAGMENT.sub(" - ", text)
    base, *tails = [part.strip() for part in re.split(r"\s+-\s+|\s+-$", text)]
    fragments += tails
    base = _INLINE_FEAT.sub("", base)

    kept = [_INLINE_FEAT.sub("", f) for f in fragments if f and _keep_fragment(f)]
    return _clean(" ".join([base] + kept))


def normalize_artist(artist):
    text = _clean(_fold(artist))
    return re.sub(r"^the ", "", text)


def base_title(title):
    """Title before any qualifier - used for fuzzy comparison."""
    text = _FRAGMENT.sub(" - ", _fold(title))
    text = re.split(r"\s+-\s+|\s+-$", text)[0]
    return _clean(_INLINE_FEAT.sub("", text))


def song_key(title, artist):
    return normalize_artist(artist), normalize_title(title)


def titles_similar(title_a, title_b):
    a, b = base_title(title_a), base_title(title_b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= FUZZY_TITLE_RATIO


def is_duplicate(title_a, artist_a, title_b, artist_b, features_a=None, features_b=None):
    """True if the two tracks look like copies of the same song (rules 1 or 2)."""
    key_a, key_b = song_key(title_a, artist_a), song_key(title_b, artist_b)
    if key_a[0] != key_b[0]:
        return False
    if key_a[1] == key_b[1]:
        return True
    if features_a is None or features_b is None:
        return False
    close = np.linalg.norm(np.asarray(features_a) - np.asarray(features_b)) < AUDIO_DUPLICATE_DISTANCE
    return close and titles_similar(title_a, title_b)


def find_duplicate_groups(titles, artists, X):
    """Assigns a group id to every track; tracks sharing an id are copies of
    one song. `X` is the standardized feature matrix (rows aligned with titles).

    Returns (group_ids, rule_counts) where rule_counts records how many merges
    each rule made, for reporting.
    """
    n = len(titles)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        parent[rj] = ri
        return True

    keys = [song_key(t, a) for t, a in zip(titles, artists, strict=True)]
    rule_counts = {"normalized_title": 0, "fuzzy_title_and_audio": 0}

    # Rule 1: identical normalized key
    first_by_key = {}
    for i, key in enumerate(keys):
        if key in first_by_key:
            rule_counts["normalized_title"] += union(first_by_key[key], i)
        else:
            first_by_key[key] = i

    # Rule 2: within each artist, near-identical audio + similar title
    by_artist = {}
    for i, (artist, _title) in enumerate(keys):
        by_artist.setdefault(artist, []).append(i)
    for rows in by_artist.values():
        if len(rows) < 2:
            continue
        sub = X[rows]
        dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        for a, b in zip(*np.nonzero(np.triu(dists < AUDIO_DUPLICATE_DISTANCE, k=1)), strict=True):
            i, j = rows[a], rows[b]
            if find(i) != find(j) and titles_similar(titles[i], titles[j]):
                rule_counts["fuzzy_title_and_audio"] += union(i, j)

    roots = [find(i) for i in range(n)]
    remap = {}
    group_ids = np.array([remap.setdefault(r, len(remap)) for r in roots])
    return group_ids, rule_counts
