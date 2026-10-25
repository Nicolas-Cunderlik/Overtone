from stemsim.registry import DONE
from stemsim.schema import PAIR, Judgment, Membership, NormalizedDataset, Track


class FakeAdapter:
    target, metric, license, url = "perceptual", "spearman", "-", "-"

    def __init__(self, name):
        self.name = name


def test_ingest_is_idempotent_and_shared_tracks_merge(registry):
    shared = Track("mtat:0/x.mp3", "mtat", audio_ref="0/x.mp3", title="X", artist="A")
    other = Track("mtat:0/y.mp3", "mtat", audio_ref="0/y.mp3")
    first = NormalizedDataset(
        tracks=[shared, other],
        members=[Membership(shared.track_id, "1"), Membership(other.track_id, "2")],
        judgments=[Judgment(PAIR, shared.track_id, other.track_id, value=0.5)],
    )
    for _ in range(2):  # re-ingesting replaces, never duplicates
        registry.replace_dataset(FakeAdapter("one"), first)
    # A second dataset knows the same clip but not its title - known fields survive
    registry.replace_dataset(
        FakeAdapter("two"),
        NormalizedDataset(
            tracks=[Track(shared.track_id, "mtat", audio_ref="0/x.mp3")], members=[Membership(shared.track_id, "a")]
        ),
    )

    db = registry.conn
    assert db.execute("SELECT COUNT(*) FROM judgments WHERE dataset = 'one'").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2
    assert db.execute("SELECT title FROM tracks WHERE track_id = ?", (shared.track_id,)).fetchone()[0] == "X"


def test_pending_skips_finished_tracks(registry):
    tracks = [Track(f"mtat:{i}", "mtat") for i in range(3)]
    registry.replace_dataset(
        FakeAdapter("one"), NormalizedDataset(tracks=tracks, members=[Membership(t.track_id, "") for t in tracks])
    )
    registry.set_status("mtat:0", "audio", DONE, artifact="audio/mtat/0.mp3")
    assert [row["track_id"] for row in registry.pending("audio")] == ["mtat:1", "mtat:2"]


def test_exclusive_refuses_a_second_holder(registry):
    import pytest

    with registry.exclusive("audio-deezer"):
        with pytest.raises(SystemExit, match="already running"), registry.exclusive("audio-deezer"):
            pass
        with registry.exclusive("audio-mtat"):  # other stages are unaffected
            pass
    with registry.exclusive("audio-deezer"):  # released on exit
        pass


def test_stage_inputs_outputs_and_windows(registry):
    tracks = [Track("mtat:a", "mtat"), Track("mtat:b", "mtat")]
    perceptual = FakeAdapter("rated")
    seeds = FakeAdapter("pool")
    seeds.target = "seeds"
    registry.replace_dataset(
        perceptual,
        NormalizedDataset(tracks=tracks, members=[Membership("mtat:a", "1", 10.0, 20.0), Membership("mtat:b", "2")]),
    )
    registry.replace_dataset(seeds, NormalizedDataset(members=[Membership("mtat:a", "x", 4.0, 24.0)]))

    # only tracks whose required stage is done are offered to the next stage
    registry.set_status("mtat:a", "audio", DONE, artifact=r"audio\mtat\a.mp3")  # stored Windows-style
    assert [r["track_id"] for r in registry.pending("separate", requires="audio")] == ["mtat:a"]

    path, _ = registry.output("mtat:a", "audio")
    assert path == registry.data_dir / "audio" / "mtat" / "a.mp3"
    assert registry.output("mtat:b", "audio") is None

    assert registry.clip_windows("mtat:a") == [(4.0, 24.0), (10.0, 20.0)]  # every dataset's judged excerpt
    assert registry.keeps_stems("rated") and not registry.keeps_stems("pool")
