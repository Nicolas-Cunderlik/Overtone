# Overtone - agent guide

PyQt5 desktop panel (Windows-only) docked beside the Spotify app. For the current track it shows the album cover, LLM chord progressions (exported as MIDI) and kNN similar tracks. README.md covers setup; this file covers how to work in the code.

## Working agreement (read first)

The owner steers this project and wants to understand it deeply, not to have it built fast. Default to **propose, explain, wait**: describe the options, the trade-offs and your recommendation, then let the owner decide. Smaller steps, each explained, beat large batches of unreviewed change.

**Ask before doing any of these**, even when the next step seems obvious:
- Design and architecture: new modules, stages, packages or scripts; schema or data-format changes; storage policy; choosing between approaches.
- Experiment design: which datasets fit vs. stay held out, metrics, baselines, hyperparameters, what counts as success.
- Dependencies and tools: installing, upgrading or removing packages, in any venv.
- Long-running or costly work: starting, stopping or restarting pipeline jobs, GPU runs, crawls or downloads, or anything taking more than a few minutes or ~1 GB of disk.
- Deleting or overwriting data (registry rows, raw files, audio, stems, embeddings), even to repair corruption. Report the problem and propose the repair.
- Anything outside the repo: GitHub issues, comments or closing them, pushes, external API use beyond agreed limits.
- Changes beyond what was asked ("while I was there" fixes): list them as suggestions instead.

**Fine without asking:** reading code, data and logs; read-only status checks; running tests and linters; making exactly the change requested, within its stated scope; fixing a lint or test failure introduced by that change.

**When something unexpected happens** (a crash, a surprising result, a data problem): stop, explain what happened and why, and propose next steps. Don't fix it silently, especially if the fix touches anything in the "ask first" list.

## Commands

| Task | Command |
|---|---|
| Run app | `overtone` (or `python -m overtone`) |
| Install / update the environment | `uv sync` (after editing any `pyproject.toml`: `uv lock`, then `uv sync`) |
| Headless smoke check | `overtone --self-test` (exit 0 = the stem model loads and a catalog lookup works) |
| Tests | `pytest` (unit only: `pytest tests/unit`) |
| Lint + format | `pre-commit run --all-files` (ruff check/format + hygiene hooks) |
| Type check | `pyright` (basic mode over `src/` and `packages/core/src/`, config in `pyproject.toml`; also runs in CI) |
| Build the catalog stem index | `.venv\Scripts\python -m stemsim index --weights-run <MLflow run id>` (from `research/stem_similarity`), then move `STEM_MODEL_VERSION` in `overtone_core/ml/paths.py` |
| Train the feature model (evaluation baseline) | `python -m overtone_core.ml.download_dataset` then `python -m overtone_core.ml.train`; compare variants with `python -m overtone_core.ml.evaluate` |
| Run CI checks locally | `python scripts/run_checks.py` |
| Research pipeline overview | `python scripts/pipeline_status.py` |
| Restart pipeline stages (background, no duplicates) | `python scripts/resume_pipeline.py [--gpu \| --audio \| --dry-run]` |
| MLflow UI / ABX rating app | `python scripts/launch_mlflow.py` / `python scripts/launch_rating_server.py` |
| Build the next ABX rating session | `.venv\Scripts\python -m stemsim session` (from `research/stem_similarity`) |

`scripts/` are standard-library launchers: run them with any Python, they use the repo's venv themselves.

One venv for everything: `.venv` at the repo root, built by `uv sync` from `uv.lock`, with the app, core and research packages installed editable plus torch (CUDA build on Windows, CPU on Linux CI), Demucs and MuQ.

## Architecture

- **Packages** (one uv workspace, one `uv.lock`):
  - `src/overtone/`: the PyQt app (UI, workers, services). Module paths below without a prefix are relative to it.
  - `packages/core/src/overtone_core/`: code the app and research both use (`ml/`, `matching.py`, `clients/`, `secrets.py`, `config.py`).
  - `research/stem_similarity/stemsim/`: the research pipeline, experiments and ABX rating app.

  The app and research both import core; neither imports the other, and core imports neither. Code both need goes to core.
- **Threading:** the UI thread never does network or disk-heavy work. Each job is a `QThread` subclass in `workers/` that emits results as signals: `PlaybackPollWorker` (Spotify poll), `PlayTrackWorker`, `FetchDataWorker` (album art, AI), `SimilarTracksWorker` (similar tracks, which may need a song analysed). `SpotifyApp` drops results from superseded workers (`sender() is not self.current_worker`) and asks a superseded `SimilarTracksWorker` to stop (`requestInterruption()`).
- **Services** (`services/`) own the external clients, create them lazily, and catch and log every failure, returning `None` or `[]` and never raising into the UI. `SpotifyService` serializes calls with a lock, because Spotify rotates refresh tokens.
- **Docking:** `services/spotify_window_service.py` uses Win32 (ctypes) plus a WinEvent hook, so the window shows and hides with Spotify's. Never call `window.show()` directly.
- **AI chords:** `ai_service.py` holds the prompt (the source of truth for prompt wording). The output contract is `music/chord_schema.py`: pydantic shape, sent to OpenAI as a strict JSON schema, plus `validate_suggestions()` for musical rules, with errors fed back for up to 3 retries. Valid results are cached 24h on disk (`chord_cache.py`) to avoid re-billing.
- **Similar tracks** (`services/similarity_service.py`): the stem model. Its index (`models/stem-muq/<version>/`: per-channel MuQ-MuLan vectors of ~20.7k catalog songs, fitted weights) is built by `stemsim index` and read by `overtone_core/ml/stem_model.py` (z-scored cosine per channel, silence gating, weighted sum; per-channel scores come with every result and are logged, not shown). A catalog song uses its stored vectors. Any other song uses its Deezer preview (`overtone_core/clients/deezer.py`: ISRC, else title search; never the device's audio), analysed by `overtone_core/ml/stem_analysis.py` (Demucs + MuQ, GPU if present), with the vectors cached by track id (`services/stem_cache.py`). The research embed stage runs the same analysis, so live and catalog vectors match exactly. No fallback model: failures show a generic message and log the detail.
- **Feature model** (`overtone_core/ml/`, `feature-knn`): Spotify legacy audio features + genre-classifier probabilities, cosine kNN. No longer used by the app; kept as the evaluation baseline (`train.py` writes it, `similarity_model.py` reads it).

## Conventions

- Shared constants have one home per domain, never a catch-all `constants.py`: the app's files and settings in `config.py`; the repo root, secrets-file location and HTTP user agent/timeout in `overtone_core/config.py`; the research pipeline's in `stemsim/config.py`; UI colours and fonts in `ui/theme.py`; model names, pinned versions and model filenames in `overtone_core/ml/paths.py`; the stem model's audio contract (channel order `STEMS`/`CHANNELS`, sample rate, embedding model) in `overtone_core/ml/stems.py`. Paths are never relative to the working directory. Single-use constants go at the top of their module in UPPER_CASE.
- An external service used by more than one module gets a client in `overtone_core/clients/` (`deezer.py`, `spotify.py`) that owns its endpoints, rate limits, error codes and retries; shared HTTP plumbing is `clients/http.py`. A service with one user keeps these at the top of that module (Last.fm in `datasets/lastfm.py`) until a second user appears.
- Comments and docstrings state current behaviour, not history. A docstring tells a caller what the code does, takes and returns. A comment covers only what the code can't say: surprising edge cases, quirks (an API that reports errors as HTTP 200), the reason for a design choice, and anything that stops someone from undoing a fix. Keep them short, and accurate when the code changes.
- snake_case everywhere; `_private` helpers. Qt overrides keep Qt's names (`closeEvent`).
- Every network call sets a timeout.
- Dependencies are pinned with exact versions in the `pyproject.toml` of the package that imports them (root: the app and the `dev` group; `packages/core/`; `research/stem_similarity/`), and `uv.lock` pins the whole workspace. torch, torchaudio, Demucs, MuQ and transformers change only together with a rebuilt stem index (live vectors must match the catalog's); Dependabot ignores them.
- Ruff config lives in `pyproject.toml` (line length 120, quote style preserved). Don't reformat by hand; let pre-commit do it.

## Secrets and data

- Secrets live only in `.env` at the repo root (template: `.env.example`), unless `OVERTONE_ENV_FILE` points elsewhere. Read them with `overtone_core.secrets.get_secret(name)` (no heavy imports, loads the file lazily), never by parsing `.env` yourself. Never log, print or commit it. `logging_config.py` caps spotipy/urllib3 logging at INFO because their DEBUG output leaks the client secret and tokens.
- Runtime state (Spotify token, log, chord cache, live songs' stem vectors) goes to `config.APP_DATA_DIR` (`%LOCALAPPDATA%\Overtone`), never the repo. Tests that run app code should point `OVERTONE_DATA_DIR` at a temp dir.
- Gitignored and regenerable: `data/` (dataset), `models/` (model versions; the SSM-2 installer will download them), research audio and embeddings. The `check-added-large-files` hook enforces this.
- Spotify's audio-features endpoint is restricted for this app. Spotify-format features for songs outside the catalog (the stem model needs none) come from FreqBlog (`FREQBLOG_API_KEY`), which allows 1000 requests/month. The app-side cap is 100/day, with a permanent cache by track_id and a catalog check before any request.

## Tests

- All tests live under `tests/`. `tests/unit/` mirrors `src/overtone/`, `tests/unit/core/` mirrors `overtone_core` and `tests/unit/research/` mirrors `stemsim`: one `test_<module>.py` per module. Unit tests must not touch the network, Win32 or trained artifacts. CI runs them on Linux.
- `tests/integration/` is for anything needing artifacts, Qt event loops or the models.
- Test the behaviour a caller relies on, through the public API, not private helpers or incidental details. Prefer fewer, larger tests built from factories and shared fixtures (`conftest.py`) when that saves setup, and `pytest.mark.parametrize` for input/output tables. Keep each test readable on its own: the values its assertions depend on appear in the test, not hidden in setup.
- CI (`.github/workflows/ci.yml`) runs pre-commit and the unit tests on every PR and every push to `main`.
