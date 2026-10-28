"""SQLite registry shared by all stages.

WAL mode lets every stage read while one writes, so a network stage and a
GPU stage can run side by side. Stages coordinate only through this file:
each one asks for rows whose inputs are ready and whose own output is
missing (stage_status), does the work, and records the outcome.

Stage outcomes:
  done         output written (`artifact` points at it)
  failed       tried and failed; retried on the next run unless the cause is permanent
  unavailable  can't be done with what we have (e.g. no audio source yet)

Different stages run side by side, but two copies of the SAME stage would
fetch the same rows and write the same files - registry.exclusive(name)
refuses to start a second one.
"""

import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import config
from .schema import NormalizedDataset

DONE, FAILED, UNAVAILABLE = "done", "failed", "unavailable"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name TEXT PRIMARY KEY, target TEXT, metric TEXT, license TEXT, url TEXT,
    ingested_at REAL, n_tracks INTEGER, n_judgments INTEGER
);
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY, audio_source TEXT NOT NULL, audio_ref TEXT NOT NULL DEFAULT '',
    title TEXT, artist TEXT, meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS memberships (
    dataset TEXT NOT NULL, track_id TEXT NOT NULL, local_id TEXT, clip_start REAL, clip_end REAL,
    PRIMARY KEY (dataset, track_id)
);
CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, kind TEXT NOT NULL,
    a TEXT NOT NULL, b TEXT NOT NULL, c TEXT, value REAL, weight REAL, grp TEXT,
    facet TEXT NOT NULL DEFAULT 'overall', meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS judgments_dataset ON judgments (dataset);
CREATE TABLE IF NOT EXISTS stage_status (
    track_id TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL,
    detail TEXT, artifact TEXT, updated_at REAL,
    PRIMARY KEY (track_id, stage)
);
"""


class Registry:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.data_dir / config.DB_FILE, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self):
        """Columns added after a registry was first created."""
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(judgments)")}
        with self.transaction() as db:
            if "facet" not in columns:
                db.execute("ALTER TABLE judgments ADD COLUMN facet TEXT NOT NULL DEFAULT 'overall'")
            if "meta" not in columns:
                db.execute("ALTER TABLE judgments ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'")

    def raw_dir(self, dataset):
        path = self.data_dir / "raw" / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audio_dir(self, source):
        path = self.data_dir / "audio" / source
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def exclusive(self, name):
        """Holds an OS file lock for the duration, so only one process runs
        `name` at a time. The OS releases it when the process exits, even on a
        crash, so a stale lock can never block a later run."""
        path = self.data_dir / "locks" / f"{name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+")  # noqa: SIM115 - held open for the lock's lifetime
        try:
            _lock(handle)
        except OSError:
            handle.close()
            raise SystemExit(f"Another process is already running {name!r} - not starting a second one") from None
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            _unlock(handle)
            handle.close()

    @contextmanager
    def transaction(self):
        with self.conn:
            yield self.conn

    # --- datasets -----------------------------------------------------------

    def replace_dataset(self, adapter, normalized: NormalizedDataset):
        """Writes an adapter's normalized output. Idempotent: the dataset's
        memberships and judgments are replaced wholesale. Tracks are shared,
        so they're merged - known fields are never overwritten with blanks."""
        with self.transaction() as db:
            db.executemany(
                """INSERT INTO tracks (track_id, audio_source, audio_ref, title, artist, meta)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (track_id) DO UPDATE SET
                     audio_ref = CASE WHEN excluded.audio_ref != '' THEN excluded.audio_ref ELSE audio_ref END,
                     title = COALESCE(title, excluded.title),
                     artist = COALESCE(artist, excluded.artist),
                     meta = json_patch(excluded.meta, meta)""",
                [
                    (t.track_id, t.audio_source, t.audio_ref, t.title, t.artist, json.dumps(t.meta))
                    for t in normalized.tracks
                ],
            )
            db.execute("DELETE FROM memberships WHERE dataset = ?", (adapter.name,))
            db.executemany(
                "INSERT OR REPLACE INTO memberships VALUES (?, ?, ?, ?, ?)",
                [(adapter.name, m.track_id, m.local_id, m.clip_start, m.clip_end) for m in normalized.members],
            )
            db.execute("DELETE FROM judgments WHERE dataset = ?", (adapter.name,))
            db.executemany(
                "INSERT INTO judgments (dataset, kind, a, b, c, value, weight, grp, facet, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (adapter.name, j.kind, j.a, j.b, j.c, j.value, j.weight, j.group, j.facet, json.dumps(j.meta))
                    for j in normalized.judgments
                ],
            )
            db.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    adapter.name,
                    adapter.target,
                    adapter.metric,
                    adapter.license,
                    adapter.url,
                    time.time(),
                    len(normalized.members),
                    len(normalized.judgments),
                ),
            )

    def dataset_tracks(self, dataset):
        return self.conn.execute(
            "SELECT t.* FROM tracks t JOIN memberships m USING (track_id) WHERE m.dataset = ? ORDER BY t.track_id",
            (dataset,),
        ).fetchall()

    # --- stage bookkeeping --------------------------------------------------

    def pending(
        self, stage, audio_source=None, dataset=None, retry_failed=True, limit=None, requires=None, skip_done=None
    ):
        """Tracks with no final outcome for `stage` yet - and, with `requires`,
        only those whose `requires` stage is done (its output is the input);
        with `skip_done`, not those whose `skip_done` stage is already done
        (`stage` only helps that one)."""
        final = (DONE, UNAVAILABLE) if retry_failed else (DONE, UNAVAILABLE, FAILED)
        placeholders = ",".join("?" * len(final))
        sql = f"""SELECT t.* FROM tracks t
                  WHERE NOT EXISTS (SELECT 1 FROM stage_status s WHERE s.track_id = t.track_id
                                    AND s.stage = ? AND s.state IN ({placeholders}))"""  # noqa: S608 - placeholders only
        args = [stage, *final]
        if requires:
            sql += """ AND EXISTS (SELECT 1 FROM stage_status r WHERE r.track_id = t.track_id
                                   AND r.stage = ? AND r.state = 'done')"""
            args.append(requires)
        if skip_done:
            sql += """ AND NOT EXISTS (SELECT 1 FROM stage_status d WHERE d.track_id = t.track_id
                                       AND d.stage = ? AND d.state = 'done')"""
            args.append(skip_done)
        if audio_source:
            sql += " AND t.audio_source = ?"
            args.append(audio_source)
        if dataset:
            sql += " AND EXISTS (SELECT 1 FROM memberships m WHERE m.track_id = t.track_id AND m.dataset = ?)"
            args.append(dataset)
        sql += " ORDER BY t.track_id"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return self.conn.execute(sql, args).fetchall()

    def output(self, track_id, stage):
        """(path, detail) of a finished stage's output for a track, or None.
        Artifacts are stored relative to the data dir (backslashes on Windows)."""
        row = self.conn.execute(
            "SELECT artifact, detail FROM stage_status WHERE track_id = ? AND stage = ? AND state = 'done'",
            (track_id, stage),
        ).fetchone()
        if row is None or row["artifact"] is None:
            return None
        return self.data_dir / Path(row["artifact"].replace("\\", "/")), row["detail"]

    def status(self, track_id, stage):
        """(state, detail) of a track's outcome for `stage`, or None if the stage hasn't run on it."""
        row = self.conn.execute(
            "SELECT state, detail FROM stage_status WHERE track_id = ? AND stage = ?", (track_id, stage)
        ).fetchone()
        return None if row is None else (row["state"], row["detail"])

    def keeps_stems(self, dataset):
        """Stems stay on disk for perceptual datasets (see stages/separate.py)."""
        row = self.conn.execute("SELECT target FROM datasets WHERE name = ?", (dataset,)).fetchone()
        return row is not None and row["target"] == "perceptual"

    def clip_windows(self, track_id):
        """The excerpts (start, end) seconds that any dataset's listeners judged for this track."""
        rows = self.conn.execute(
            """SELECT DISTINCT clip_start, clip_end FROM memberships
               WHERE track_id = ? AND clip_start IS NOT NULL AND clip_end IS NOT NULL""",
            (track_id,),
        ).fetchall()
        return sorted((row["clip_start"], row["clip_end"]) for row in rows)

    def set_status(self, track_id, stage, state, detail=None, artifact=None):
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO stage_status VALUES (?, ?, ?, ?, ?, ?)",
                (track_id, stage, state, detail, artifact, time.time()),
            )

    def summary(self, stages):
        """Per dataset: target, metric, tracks, judgments, then for each of
        `stages` how many of its tracks are done / failed / unavailable."""
        datasets = self.conn.execute(
            "SELECT name, target, metric, n_tracks, n_judgments FROM datasets ORDER BY name"
        ).fetchall()
        counts = {
            (row["dataset"], row["stage"], row["state"]): row["n"]
            for row in self.conn.execute(
                f"""SELECT m.dataset, s.stage, s.state, COUNT(*) AS n
                    FROM memberships m JOIN stage_status s ON s.track_id = m.track_id
                    WHERE s.stage IN ({",".join("?" * len(stages))})
                    GROUP BY m.dataset, s.stage, s.state""",  # noqa: S608 - placeholders only
                list(stages),
            )
        }
        return [
            (
                dict(row),
                {
                    stage: {state: counts.get((row["name"], stage, state), 0) for state in (DONE, FAILED, UNAVAILABLE)}
                    for stage in stages
                },
            )
            for row in datasets
        ]


if sys.platform == "win32":
    import msvcrt

    def _lock(handle):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle):
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle):
        fcntl.flock(handle, fcntl.LOCK_UN)
