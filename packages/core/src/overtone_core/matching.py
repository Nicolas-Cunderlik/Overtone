"""Whether a search hit (Deezer, Spotify) is the recording that was asked for,
not another version of it. Built on the song-identity rules in ml.dedup."""

from .ml import dedup

# Search hits further than this from the known duration are another cut
DURATION_TOLERANCE_S = 12
# Max duration gap for accepting a title that only adds plain words; 12 s let in an acoustic cut
EXTRA_WORDS_DURATION_S = 2


def isrc_match_plausible(title, artist, hit_title, hit_artists):
    """Whether a hit found by ISRC is this track: its artist or its title must
    match. Requiring both would drop real matches (romanised titles, renamed artists)."""
    want = dedup.normalize_artist(artist or "")
    return any(dedup.normalize_artist(name) == want for name in hit_artists) or dedup.titles_similar(
        title or "", hit_title
    )


def matching_hits(title, artist, duration, hits, title_of, artists_of, duration_of):
    """Hits by the same artist, within DURATION_TOLERANCE_S of a known
    duration, that pass same_recording; exact title matches first."""
    want_artist = dedup.normalize_artist(artist)
    wanted = dedup.normalize_title(title)
    candidates = [
        hit
        for hit in hits
        if any(dedup.normalize_artist(name) == want_artist for name in artists_of(hit))
        and not (duration and abs(duration_of(hit) - duration) > DURATION_TOLERANCE_S)
    ]
    for hit in sorted(candidates, key=lambda hit: dedup.normalize_title(title_of(hit)) != wanted):
        close = bool(duration) and abs(duration_of(hit) - duration) <= EXTRA_WORDS_DURATION_S
        if same_recording(title, title_of(hit), durations_match=close):
            yield hit


def same_recording(wanted_title, found_title, durations_match=False):
    """Whether a search hit is the recording we asked for, not another version.

    Equal after the app's title normalization (packaging like "feat." or
    "Remastered" dropped) is a match. Beyond that, a fuzzy match (typos,
    punctuation) is only trusted between two plain titles - if either names a
    distinct recording ("Remix", "Live", "Acoustic", "Extended", or anything
    unrecognised), it must match exactly: fuzzy matching alone accepts
    another remix, the live cut or the extended mix.

    With `durations_match`, a title that is the other plus plain words also
    matches ("C.R.E.A.M." vs "C.R.E.A.M. (Cash Rules Everything Around Me)")."""
    wanted, found = dedup.normalize_title(wanted_title), dedup.normalize_title(found_title)
    if wanted == found:
        return True
    if _version_words(wanted) != _version_words(found):
        return False  # e.g. "I Wanna Rock" vs "I Wanna Rock Nasty Mix" - no brackets to split on
    if durations_match and _adds_only_plain_words(wanted_title, found_title):
        return True
    plain = wanted == dedup.base_title(wanted_title) and found == dedup.base_title(found_title)
    return plain and dedup.titles_similar(wanted_title, found_title)


def _adds_only_plain_words(title_a, title_b):
    """Whether one title is the other plus words that don't name a version."""
    if dedup.base_title(title_a) != dedup.base_title(title_b):
        return False
    shorter, longer = sorted((dedup.normalize_title(title_a).split(), dedup.normalize_title(title_b).split()), key=len)
    extra, matched = [], 0
    for word in longer:
        if matched < len(shorter) and word == shorter[matched]:
            matched += 1
        else:
            extra.append(word)
    return matched == len(shorter) and not _version_words(" ".join(extra))


def _version_words(normalized_title):
    """Words marking a distinct recording (remix, live, acoustic...) anywhere in the title."""
    return {m.group(0) for m in dedup._DISTINCT.finditer(dedup._CANONICAL_MIX.sub("", normalized_title))}
