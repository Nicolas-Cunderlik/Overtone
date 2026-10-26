"""stemsim.rating.ratings: the rating format and song splits."""

import pytest
from stemsim import config
from stemsim.rating import ratings


def row(**changes):
    base = {
        "v": 2,
        "id": "s1-01",
        "rater_id": "nicolas",
        "session": "s1",
        "design": ratings.NEIGHBOURHOOD,
        "split": ratings.FIT,
        "x": "song:x",
        "a": "song:a",
        "b": "song:b",
        "play": "mix",
        "audio": {clip: {"source": "deezer", "ref": f"audio/deezer/{clip}.mp3"} for clip in ratings.CLIPS},
        "stems": None,
        "window": None,
        "repeat_of": None,
        "selection": {"rule": ratings.NEIGHBOURHOOD},
        "choice": ratings.A,
        "strength": ratings.CLEARLY,
        "strength_source": "rater",
        "tags": ["bass"],
        "note": "",
        "bad_clip": None,
        "timing": {"total_ms": 30000},
        "rated_at": "2026-10-25T20:00:00",
    }
    return base | changes


def test_song_split_is_stable_and_near_the_target_share():
    ids = [f"song:artist {i}|title {i}" for i in range(5000)]
    first = [ratings.song_split(t) for t in ids]
    assert first == [ratings.song_split(t) for t in ids]  # no state: same id, same side, forever
    share = first.count(ratings.VALIDATION) / len(ids)
    assert abs(share - ratings.VALIDATION_PERCENT / 100) < 0.02


def test_always_validation_overrides_the_hash():
    fit_song = next(t for t in (f"song:{i}" for i in range(100)) if ratings.song_split(t) == ratings.FIT)
    assert ratings.song_split(fit_song, always_validation={fit_song}) == ratings.VALIDATION


def test_triplet_split_is_none_when_songs_straddle_it():
    ids = [f"song:{i}" for i in range(200)]
    fit = [t for t in ids if ratings.song_split(t) == ratings.FIT][:3]
    val = [t for t in ids if ratings.song_split(t) == ratings.VALIDATION][:3]
    assert ratings.triplet_split(fit) == ratings.FIT
    assert ratings.triplet_split(val) == ratings.VALIDATION
    assert ratings.triplet_split([*fit[:2], val[0]]) is None


def test_check_accepts_valid_rows():
    assert ratings.check(row()) == []
    assert ratings.check(row(choice=ratings.TIE, strength=None, strength_source=None)) == []
    bad = {"clips": ["b"], "reason": "dialogue"}
    assert ratings.check(row(choice=ratings.BAD_CLIP, strength=None, strength_source=None, bad_clip=bad)) == []
    assert ratings.check(row(design=ratings.PILOT, session="pilot")) == []


def test_check_wants_exactly_the_format_fields():
    extra, missing = row(legacy_id=0), row()
    del missing["timing"]
    assert any("unknown ['legacy_id']" in p for p in ratings.check(extra))
    assert any("missing ['timing']" in p for p in ratings.check(missing))


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"strength": None, "strength_source": None}, "needs strength"),
        ({"strength": 3}, "needs strength"),
        ({"design": ratings.PILOT, "strength": None, "strength_source": None}, "needs strength"),
        ({"strength_source": None}, "strength_source"),
        ({"choice": ratings.TIE, "strength_source": None}, "takes no strength"),
        ({"choice": ratings.BAD_CLIP, "strength": None, "strength_source": None}, "bad_clip needs clips"),
        ({"bad_clip": {"clips": ["b"], "reason": "dialogue"}}, "only set when"),
        ({"tags": ["groove/tempo"]}, "unknown tags"),
        ({"split": "test"}, "split must be"),
        ({"design": ratings.REPEAT}, "repeat_of"),
        ({"rater_id": ""}, "rater_id"),
        ({"audio": {}}, "audio needs a source"),
        ({"window": [20, 10]}, "window must be"),
    ],
)
def test_check_names_each_problem(changes, expected):
    assert any(expected in problem for problem in ratings.check(row(**changes)))


def test_append_refuses_invalid_rows(tmp_path):
    path = tmp_path / "ratings.jsonl"
    ratings.append(path, row())
    with pytest.raises(ValueError, match="needs strength"):
        ratings.append(path, row(id="s1-02", strength=None, strength_source=None))
    assert [r["id"] for r in ratings.load_jsonl(path)] == ["s1-01"]


def test_flagged_tracks():
    bad = {"clips": ["a", "b"], "reason": "silence"}
    rows = [row(), row(choice=ratings.BAD_CLIP, strength=None, strength_source=None, bad_clip=bad)]
    assert ratings.flagged_tracks(rows) == {"song:a": "silence", "song:b": "silence"}


def test_every_checked_in_rating_is_valid():
    rows = ratings.load_jsonl(config.RATINGS_FILE)
    assert rows
    assert {r["id"]: ratings.check(r) for r in rows if ratings.check(r)} == {}
    assert len({r["id"] for r in rows}) == len(rows)
    assert {tuple(r) for r in rows} == {ratings.FIELDS}  # one schema, one field order
