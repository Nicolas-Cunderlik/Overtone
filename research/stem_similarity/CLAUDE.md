# stemsim - research pipeline guide

The stem-similarity study: it gathers human similarity judgments from several datasets, gets each song's audio, splits it into stems, embeds the mix and each stem, fits per-stem weights against the judgments, and builds the catalog index the app ships. The root `CLAUDE.md` (working agreement, repo conventions) applies here too. This file is the map; each module's docstring has the detail and is the source of truth.

`stemsim` imports `overtone_core`, never the app. The analysis that turns audio into vectors (`overtone_core.ml.stem_analysis`: Demucs + MuQ-MuLan) is the same code the app runs on a live song, so catalog and live vectors match. That is why torch, Demucs, MuQ and transformers only change together with a rebuilt index.

## Running it

One venv for everything: the repo root's `.venv` (`uv sync` at the repo root). Run `stemsim` from `research/stem_similarity` with that venv active. Every path is resolved from the package, not the working directory; the exception is a relative `STEMSIM_DATA_DIR`.

| Task | Command |
|---|---|
| All commands and options | `python -m stemsim --help` (the list with examples is in `stemsim/__main__.py`) |
| Register a dataset | `python -m stemsim ingest <dataset>` (`datasets` lists them) |
| Run a stage | `python -m stemsim isrc \| audio \| separate \| embed [--dataset D] [--limit N]` |
| Progress per dataset and stage | `python -m stemsim status`, or `python scripts/pipeline_status.py` (also what's running, disk, log tails) |
| Restart the long stages in the background | `python scripts/resume_pipeline.py [--gpu \| --audio \| --dry-run]` (logs in `data/logs/`) |
| Fit the stem weights | `python -m stemsim.experiments.stem_weights` (one MLflow run) |
| Add findings to a run's report | `python -m stemsim.experiments.report RUN --findings FILE ...` |
| Browse runs | `python scripts/launch_mlflow.py` |
| Build the next index version | `python -m stemsim index --weights-run <run id>` (then see the root `CLAUDE.md`) |
| Rate | `python -m stemsim session`, then `python scripts/launch_rating_server.py` |

`scripts/` are run from the repo root with any Python; they find the venv themselves.

## Pipeline

```
ingest -> isrc -> audio -> separate -> embed -> experiments (stem_weights) -> index -> app
                                         \-> session -> rate -> ratings.jsonl -> ingest listener_abx
```

| Stage | Module | Uses | Reads -> writes |
|---|---|---|---|
| `ingest` | `stages/ingest.py` + `datasets/` | network (Last.fm: a little more each run) | raw files -> registry rows (tracks, memberships, judgments) |
| `isrc` | `stages/isrc.py` | Spotify API | tracks without audio -> ISRC in the status detail |
| `audio` | `stages/audio.py` | Deezer, or local archives | ISRC (else artist + title) -> `data/audio/<source>/` |
| `separate` | `stages/separate.py` | GPU | audio -> FLAC stems, perceptual datasets only |
| `embed` | `stages/embed.py` | GPU | stems (or in-memory separation) -> one `.npz` per track |

- **Datasets** (`datasets/`) each declare a target and a metric. Perceptual: `listener_abx` (the owner's ratings), `dim_sim`, `magnatagatune`, `ccmsim`, `matcha` (no audio yet). Behavioral: `lastfm`. Seeds (the candidate pool, no judgments): `spotify_catalog`. Results are reported per dataset and never averaged across targets.
- **Outcomes**, per track and stage, in the registry's `stage_status`: `done` (output written, `artifact` points at it), `failed` (retried next run), `unavailable` (can't be done with what we have, not retried). A rerun picks up only tracks with no `done`/`unavailable` row whose input stage is done, so every stage is safe to stop and restart.
- **Order:** `separate` and `embed` without `--dataset` work through datasets in `PRIORITY` order (`__main__.py`), evaluation data first. `isrc` and `audio` skip tracks that only `config.NO_AUDIO_DATASETS` (Last.fm) includes, unless that dataset is named.
- **Downstream of the stages:** experiments read the registry and embeddings and write only MLflow runs; `index.py` writes `models/stem-muq/vN/` at the repo root; `rating/` builds sessions and serves the ABX app.

## Running stages side by side

The registry (`data/registry.sqlite`) is the only coordination between stages. It runs in WAL mode: any number of readers alongside one writer, so a network stage and a GPU stage run at the same time.

Two copies of the same stage would fetch the same rows and write the same files, so each command holds a lock for its whole run: `registry.exclusive(name)`, an OS file lock on `data/locks/<name>.lock`. A second copy exits with "Another process is already running ...". The OS releases the lock when the process exits, even on a crash, so a leftover `.lock` file never blocks anything. Lock names:

- `ingest-<dataset>`
- `isrc`
- `audio-<source>` (one per source: `deezer`, `mtat`, `pilot`, `other`; `audio` without `--source` holds all of them)
- `separate`
- `embed-<model>` (now `embed-muq-mulan-large`)

`resume_pipeline.py` checks these locks before starting a job, so it never starts a duplicate. Different stages never need a lock against each other.

## Data layout

Everything regenerable lives in `data/` (gitignored; `STEMSIM_DATA_DIR` moves it):

```
data/
  registry.sqlite          datasets, tracks, memberships, judgments, stage_status
  raw/<dataset>/           what each adapter's fetch downloaded
  audio/<source>/          audio stage output (deezer/, mtat/)
  stems/<h[:2]>/<h>/       <stem>.flac, h = sha1 of the track id (first 20 hex)
  embeddings/<model>/      <h[:2]>/<h>.npz: windows, channels, emb (W, 5, 512), rms_db
  sessions/                rating sessions, <rater>-<nnn>.jsonl
  mlflow.db, mlartifacts/  experiment runs and their artifacts (report.md, weights.json, ...)
  models/whisper/          dictation model for the rating app
  locks/, logs/            stage locks; resume_pipeline.py job logs
  backups/                 registry rows saved before a bulk change
```

Kept in this folder, outside `data/`:

- `ratings.jsonl` (committed): every ABX answer, append-only. It can't be regenerated, so never rewrite or reorder it; format in `rating/ratings.py`.
- `pilot_songs.csv` (committed) and `audio/` (gitignored): the pilot's songs and the previews that were actually rated. The `listener_abx` adapter and the `pilot` audio source still read them.
- `investigation_and_plan.txt`: the study's motivation, targets and original plan. Where it and the code disagree, the code wins.

## Conventions

- **Adding a dataset:** subclass `DatasetAdapter` (`datasets/base.py`) and add it to `ADAPTERS`. `fetch` may use the network and must be safe to rerun; `normalize` is pure (raw files -> `schema.NormalizedDataset`), so it is unit-tested from fixture files. Nothing downstream changes.
- **Adding an audio source:** one resolver class in `stages/audio.py`, keyed by `Track.audio_source`.
- **Adding an embedding model:** a new `embed:<model>` stage over the same stems, never a migration of the existing vectors.
- **Writing a stage:** pick up only rows that are ready and not done, record every outcome with `set_status`, write files to a temp path and rename, and use `failed` for transient errors and `unavailable` for permanent ones. Store artifacts relative to the data dir.
- **Experiments** are read-only over the registry and outputs, log through `tracking.run()` (git commit and per-dataset data counts on every run) and attach a `report.md` (`experiments/report.py`). An experiment's module docstring is its question and approach; the report quotes it.
- **Held-out data:** `listener_abx` and `ccmsim` are never fitted on, and fit-set triplets containing a held-out song are dropped. Rating splits are per song (`rating/ratings.py`), and pilot songs are always validation.
- **Imports:** only `separate` and `embed` load torch/Demucs/MuQ, imported inside their commands, so the rest of the CLI and the rating server run without them. Stage names live in `config.py` for the same reason.
- **Constants:** shared ones in `stemsim/config.py`; the audio contract (channels, sample rate) in `overtone_core/ml/stems.py`.
- **Tests:** `tests/unit/research/` mirrors `stemsim` (one `test_<module>.py` per module). They use a registry in a temp dir (`Registry(data_dir=tmp_path)`, see `conftest.py`) and fixture files, never the network, GPU or the real data dir.
