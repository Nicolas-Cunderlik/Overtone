"""stemsim.rating.server: sessions, answers and byte ranges - no HTTP, no browser."""

import json

import pytest
from stemsim.rating import ratings, server


def session_row(row_id, play="mix", **changes):
    row = {
        "id": row_id,
        "session": "s1",
        "design": "neighbourhood",
        "split": "fit",
        "x": "song:x",
        "a": "song:a",
        "b": "song:b",
        "play": play,
        "audio": {k: {"source": "deezer", "ref": f"audio\\deezer\\{k}.mp3"} for k in "xab"},
        "stems": {k: f"stems/{k}/drums.flac" for k in "xab"} if play != "mix" else None,
        "window": None,
        "info": {k: {"title": f"Song {k}", "artist": "Artist", "year": 2001} for k in "xab"},
    }
    return row | changes


@pytest.fixture
def session(tmp_path):
    rows = [session_row("s1-01"), session_row("s1-02", play="drums"), session_row("s1-03")]
    path = tmp_path / "s1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    for k in "xab":
        for ref in (f"audio/deezer/{k}.mp3", f"stems/{k}/drums.flac"):
            (tmp_path / ref).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / ref).write_bytes(b"audio")
    return server.Session(path, "rater1", tmp_path / "ratings.jsonl", data_dir=tmp_path)


def answer(row_id, **changes):
    base = {"id": row_id, "choice": "a", "strength": 2, "tags": ["bass"], "note": "  same bass "}
    return base | {"timing": {"total_ms": 30000}} | changes


def test_state_hides_songs_and_walks_the_session(session):
    state = session.state()
    assert (state["rated"], state["total"], state["current"]["id"]) == (0, 3, "s1-01")
    assert "song:x" not in json.dumps(state)  # nothing that names a song before answering
    assert state["upcoming"]["x"] == "/audio/s1-02/x"

    reveal = session.rate(answer("s1-01"))
    assert reveal["reveal"]["x"]["title"] == "Song x"
    state = session.state()
    assert (state["rated"], state["current"]["id"], state["median_s"]) == (1, "s1-02", 30)


def test_rate_writes_a_valid_row_from_the_session(session):
    session.rate(answer("s1-01"))
    (row,) = ratings.load_jsonl(session.ratings_path)
    assert ratings.check(row) == []
    assert (row["rater_id"], row["split"], row["note"], row["strength_source"]) == (
        "rater1",
        "fit",
        "same bass",
        "rater",
    )
    assert "info" not in row


def test_rate_refuses_bad_answers_and_double_rating(session):
    with pytest.raises(ValueError, match="needs strength"):
        session.rate(answer("s1-01", strength=None))
    with pytest.raises(ValueError, match="unknown triplet"):
        session.rate(answer("nope"))
    session.rate(answer("s1-01"))
    with pytest.raises(ValueError, match="already rated"):
        session.rate(answer("s1-01"))


def test_bad_clip_answers_are_saved_and_end_the_triplet(session):
    session.rate(answer("s1-01", choice="bad_clip", strength=None, bad_clip={"clips": ["b"], "reason": "silence"}))
    assert session.state()["current"]["id"] == "s1-02"
    assert ratings.flagged_tracks(ratings.load_jsonl(session.ratings_path)) == {"song:b": "silence"}


def test_audio_file_serves_the_mix_or_the_played_stem_only(session):
    assert session.audio_file("s1-01", "x").as_posix().endswith("audio/deezer/x.mp3")
    assert session.audio_file("s1-02", "a").as_posix().endswith("stems/a/drums.flac")
    assert session.audio_file("s1-01", "../../etc") is None
    assert session.audio_file("missing", "x") is None


def test_consistency_compares_direction_across_a_swap():
    rows = [
        {"id": "1", "choice": "a", "repeat_of": None},
        {"id": "2", "choice": "b", "repeat_of": "1"},  # swapped A/B, same opinion
        {"id": "3", "choice": "a", "repeat_of": None},
        {"id": "4", "choice": "a", "repeat_of": "3"},  # changed its mind
    ]
    assert server.consistency(rows) == {"consistent": 1, "checked": 2}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=0-", (0, 99)),
        ("bytes=10-19", (10, 19)),
        ("bytes=-10", (90, 99)),
        ("bytes=50-500", (50, 99)),
    ],
)
def test_byte_range(header, expected):
    assert server.byte_range(header, 100) == expected


def test_byte_range_rejects_what_cannot_be_served():
    with pytest.raises(ValueError):
        server.byte_range("bytes=200-", 100)
    assert server.byte_range("bytes=0-1,5-6", 100) is None  # multi-range: send the whole file


def test_page_reads_only_config_keys_the_server_sends():
    import re

    used = set(re.findall(r"CFG\.(\w+)", (server.PAGE).read_text(encoding="utf-8")))
    assert used and used <= set(server.PAGE_CONFIG)
