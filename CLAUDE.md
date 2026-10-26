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
| Headless smoke check | `overtone --self-test` (exit 0 = model loads and a lookup works) |
| Tests | `pytest` (unit only: `pytest tests/unit`) |
| Lint + format | `pre-commit run --all-files` (ruff check/format + hygiene hooks) |
| Train similarity model | `python -m overtone.ml.download_dataset` then `python -m overtone.ml.train` |
| Compare model variants | `python -m overtone.ml.evaluate` |
| Build exe | `pyinstaller app.spec`, then `dist\Overtone.exe --self-test` |
| Run CI checks locally | `python scripts/run_checks.py` |
| Research pipeline overview | `python scripts/pipeline_status.py` |
| Restart pipeline stages (background, no duplicates) | `python scripts/resume_pipeline.py [--gpu \| --audio \| --dry-run]` |
| MLflow UI / ABX rating app | `python scripts/launch_mlflow.py` / `python scripts/launch_rating_server.py` |
| Build the next ABX rating session | `.venv\Scripts\python -m stemsim session` (from `research/stem_similarity`) |

`scripts/` are standard-library launchers: run them with any Python, they find the right venv themselves.

Use the project venv (`.venv\Scripts\python`), where the package is installed editable (`pip install -e ".[dev]"`). Code lives in `src/overtone/`; module paths below are relative to it. `research/stem_similarity/` has its own venv (`research/stem_similarity/.venv`, torch + Demucs + MuQ).

## Architecture

- **Threading:** the UI thread never does network or disk-heavy work. Each job is a `QThread` subclass in `workers/` that emits results as signals: `PlaybackPollWorker` (Spotify poll), `PlayTrackWorker`, `FetchDataWorker` (album art, similar tracks, AI). `SpotifyApp` drops results from superseded workers (`sender() is not self.current_worker`).
- **Services** (`services/`) own the external clients, create them lazily, and catch and log every failure, returning `None` or `[]` and never raising into the UI. `SpotifyService` serializes calls with a lock, because Spotify rotates refresh tokens.
- **Docking:** `services/spotify_window_service.py` uses Win32 (ctypes) plus a WinEvent hook, so the window shows and hides with Spotify's. Never call `window.show()` directly.
- **AI chords:** `ai_service.py` holds the prompt (the source of truth for prompt wording). The output contract is `music/chord_schema.py`: pydantic shape, sent to OpenAI as a strict JSON schema, plus `validate_suggestions()` for musical rules, with errors fed back for up to 3 retries. Valid results are cached 24h on disk (`chord_cache.py`) to avoid re-billing.
- **Similarity model** (`ml/`): Spotify legacy audio features → standardized, key as sin/cos at low weight → plus genre-classifier probabilities → cosine kNN over a deduplicated 25.6k-song catalog. `train.py` writes the artifacts; `similarity_model.py` reads them; filenames are defined in `ml/paths.py`. `find_similar_to_features()` serves songs outside the catalog (FreqBlog features).

## Conventions

- Shared constants have one home: file locations and settings in `config.py` (never relative to the working directory; it handles the PyInstaller frozen case), UI colours and fonts in `ui/theme.py`, ML file locations in `ml/paths.py`. Single-use constants go at the top of their module in UPPER_CASE.
- Module docstrings explain *why* (design decisions, measured trade-offs, numbers from evaluate.py). Keep them current when behaviour changes, and match the existing comment density.
- snake_case everywhere; `_private` helpers. Qt overrides keep Qt's names (`closeEvent`).
- Every network call sets a timeout.
- Dependencies are pinned in `pyproject.toml`: `dependencies` (app) and the `dev` extra (tooling). Add new ones there with exact versions. The research venv has its own `research/stem_similarity/requirements.txt`.
- Ruff config lives in `pyproject.toml` (line length 120, quote style preserved). Don't reformat by hand; let pre-commit do it.

## Secrets and data

- Secrets live only in `.env` at the repo root (template: `.env.example`), or next to the exe when frozen. Read them with `overtone.secrets.get_secret(name)` (no heavy imports, loads the file lazily), never by parsing `.env` yourself. Never log, print or commit it. `logging_config.py` caps spotipy/urllib3 logging at INFO because their DEBUG output leaks the client secret and tokens.
- Runtime state (Spotify token, log, chord cache) goes to `config.APP_DATA_DIR` (`%LOCALAPPDATA%\Overtone`), never the repo. Tests that run app code should point `OVERTONE_DATA_DIR` at a temp dir.
- Gitignored and regenerable: `data/` (dataset), `models/` (trained artifacts, bundled into the exe), research audio and embeddings. The `check-added-large-files` hook enforces this.
- Spotify's audio-features endpoint is restricted for this app. Features for new songs come from FreqBlog (`FREQBLOG_API_KEY`), which allows 1000 requests/month. The app-side cap is 100/day, with a permanent cache by track_id and a catalog check before any request.

## Tests

- All tests live under `tests/`. `tests/unit/` mirrors `src/overtone/`: one `test_<module>.py` per module. The research pipeline's (`research/stem_similarity/stemsim`) unit tests are in `tests/unit/research/`. Unit tests must not touch the network, Win32 or trained artifacts. CI runs them on Linux.
- `tests/integration/` is for anything needing artifacts, Qt event loops or the built exe.
- CI (`.github/workflows/ci.yml`) runs pre-commit and the unit tests on every PR and every push to `main`.
