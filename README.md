# Overtone

[Jira board](https://nicolascunderlik.atlassian.net/jira/software/c/projects/SSM/list)

A desktop panel for music producers that sits beside the Spotify app and follows whatever is playing: chord progressions in the style of the current song, exported as MIDI, and similar songs found by how they actually sound.

**[Download for Windows](https://github.com/Nicolas-Cunderlik/Overtone/releases/latest)**, then follow [Install](#install).

## Why

Streaming recommenders lean on listening behaviour (what people play together) and whole-mix descriptors like energy or danceability. That favours songs that are already popular, and it can't say *which part* of two songs is alike: a folk vocal over a trap beat matches neither folk nor trap well.

Overtone recommends from the audio alone. It splits each song into stems (drums, bass, vocals, other), embeds each one, and learns how to weigh them from human similarity judgments, including a producer's own ratings. No play counts are involved, so a new or niche song competes on equal terms with a hit, and per-stem scores make recommendations explainable ("similar bass") and, eventually, steerable.

## Architecture

```mermaid
flowchart LR
    subgraph research["Research pipeline - research/stem_similarity"]
        direction TB
        JUDGE["Human similarity judgments<br/>DIM-SIM, MagnaTagATune, CCMSim,<br/>MATCHA, Last.fm, own ABX ratings"]
        RATE["ABX rating app"]
        AUDIO["Audio previews<br/>Deezer"]
        SEP["Stem separation<br/>Demucs: drums, bass, vocals, other"]
        EMB["Embeddings<br/>MuQ-MuLan: mix + each stem"]
        FIT["Fit + evaluate stem weights<br/>bootstrap CIs, MLflow runs + reports"]
        RATE --> JUDGE
        AUDIO --> SEP --> EMB --> FIT
        JUDGE --> FIT
    end

    subgraph app["Desktop app - src/overtone"]
        direction TB
        PLAY["Spotify playback<br/>polling + window docking"]
        COVER["Album cover"]
        CHORDS["Chord progressions<br/>validated, exported as MIDI"]
        SIM["Similar tracks"]
        PLAY --> COVER
        PLAY --> CHORDS
        PLAY --> SIM
    end

    LLM["OpenAI"] --> CHORDS
    FIT --> INDEX["Catalog stem index<br/>+ fitted weights"]
    INDEX --> SIM
    PLAY --> LIVE["Live analysis of the current song<br/>Deezer preview, stems, embeddings"]
    LIVE --> SIM
```

The research pipeline and the app share one package, `overtone_core` (`packages/core`): the Deezer lookup and the stem analysis that build the catalog index are the same code that analyses the song playing now, so the two are directly comparable.

## Features

- **Docked panel**: sits to the left of the Spotify desktop app, shows and hides with its window, and follows playback.
- **Chord progressions**: three progressions in the style of the current song, checked against music-theory rules (the model is re-asked when they fail) and cached for a day. Click one to save it as a MIDI file to Downloads.
- **Similar tracks**: the three closest songs from a ~20.7k-song catalog by how they sound: the mix and each stem compared separately, weighted by fitted stem weights. A song outside the catalog is analysed from its Deezer preview (about 1.5 s on a GPU once the models are loaded) and remembered. Click a result to play it (Spotify Premium) or open it in Spotify.
- **Album cover view**: click the cover to scale it to your display (Esc or Space closes it).
- **Responsive UI**: all Spotify, OpenAI and network calls run on worker threads.

## Research status

The stem-similarity study (`research/stem_similarity`) has separated and embedded about 30k tracks and fits per-stem weights against human judgments. So far:

- Combining stems agrees with listeners more than the mix alone on DIM-SIM: +2.7 points (95% CI +1.1 to +4.3). On MagnaTagATune and the held-out sets no gain is shown yet.
- The MuQ-MuLan mix embedding already beats the app's current feature model on the author's ABX ratings: +26.7 points (95% CI +6.7 to +53.3, 15 ratings).
- Each stem alone is weaker than the mix, and learned weights don't yet beat equal weights. More ABX ratings are the main bottleneck.

## Roadmap

- **Update check**: tell the user when a newer version is out (app and model versions move together).
- **Hosted backend**: an Overtone server that holds the OpenAI key and runs the model, so the app becomes a small client; local first, then in the cloud.
- **Faster ABX ratings**: a rating app that picks the most informative comparisons, in short sessions.
- **Explanations and steering**: "recommended because: similar bass", and requests like "like this, but match the drums".
- **Personal taste**: a producer-specific similarity layered on the population model.
- **Better per-stem embeddings**: test how well the embedding model handles isolated stems; fine-tune or switch models if needed.
- **In-app feedback**: the thumbs up/down box on recommendations is in place but not wired yet.
- **Beyond Spotify**: other playback providers.

## Requirements

- Windows 10 or 11, 64-bit (the panel docks to Spotify's window through the Win32 API), and the Spotify desktop app
- A Spotify developer app (client ID/secret, redirect URI `http://127.0.0.1:<port>`) and, for chord suggestions, an OpenAI API key
- About 10 GB of disk space: Python, torch, the model weights and the catalog index
- An NVIDIA GPU makes analysing songs outside the catalog fast (about 1.5 s per song on an RTX 3070); without one it runs on the CPU (about 35 s per song)

## Install

You don't need Python, git or any code. You need the Spotify desktop app and about 10 GB of free disk space.

**1. Make your Spotify keys** (about 5 minutes, once). Overtone reads what's playing through your own Spotify developer app:

1. Log in at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) with your Spotify account and click **Create app**.
2. Give it any name and description. Under **Redirect URIs** add `http://127.0.0.1:4000`, tick **Web API**, accept the terms and save.
3. In the app's **Settings**, copy the **Client ID**, then click **View client secret** and copy that too.

Optional: an [OpenAI API key](https://platform.openai.com/api-keys) for the chord suggestions. Without one, everything else still works.

**2. Run the installer.** Download `Overtone-setup-<version>.exe` from the [latest release](https://github.com/Nicolas-Cunderlik/Overtone/releases/latest) and run it. Windows SmartScreen warns because the installer isn't signed: click **More info**, then **Run anyway**. When it asks, paste your keys. Then setup downloads the rest, which takes a while (about 10 GB on a first install):

1. Python 3.13 and the exact packages Overtone was tested with (torch, Demucs, MuQ...), checked against their hashes.
2. The catalog index (about 100 MB), checked against the hash pinned in the code.
3. The Demucs and MuQ-MuLan model weights (about 4 GB), followed by a quick self-test.

**3. Start Overtone** from the Start menu while Spotify is open. The first time, a browser window asks you to allow access to your Spotify account.

Everything goes into `%LOCALAPPDATA%\Programs\Overtone`; your keys are in the `.env` file there, if you need to change them. Running a newer installer upgrades in place and keeps your keys. Uninstall from Windows Settings → Apps. If setup stops (for example, the connection drops), run it again: it continues where it stopped, and the details are in `install\install.log` in that folder.

The models' licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): the MuQ-MuLan weights are for non-commercial use only.

Maintainers: `python scripts/build_installer.py` builds the installer into `dist/`; attach it to a GitHub Release tagged `v<version>`.

## Development setup

Needs Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                           # one .venv for the app and the research pipeline, from uv.lock (~3 GB with CUDA torch)
.venv\Scripts\pre-commit install
copy .env.example .env            # then fill in the keys
```

Similar tracks need the catalog stem index in `models/stem-muq/v1/`. Download the published one (about 100 MB, checked against the hash in `paths.py`):

```bash
python -m overtone_core.ml.model_release fetch
```

Or build it with the research pipeline. That fetches about 30k Deezer previews, then separates and embeds them on a GPU, so it takes a long time. With the venv active (`.venv\Scripts\activate`):

```bash
python -m overtone_core.ml.download_dataset      # the catalog's song list, into data/
cd research/stem_similarity
python -m stemsim ingest spotify_catalog         # the candidate songs
python -m stemsim ingest dim_sim                 # the judgments the stem weights are fitted on...
python -m stemsim ingest magnatagatune --with-audio
python -m stemsim ingest listener_abx            # ...and the held-out ones
python -m stemsim ingest ccmsim
python -m stemsim isrc                           # Spotify ISRCs, so Deezer gives the exact recording
python -m stemsim audio                          # previews
python -m stemsim separate --dataset dim_sim     # stems kept on disk; repeat for magnatagatune, listener_abx, ccmsim
python -m stemsim embed                          # mix + stem vectors for every track
python -m stemsim.experiments.stem_weights       # fits the weights; the run id is in MLflow (scripts/launch_mlflow.py)
python -m stemsim index --weights-run <run id>   # writes models/stem-muq/vN/
```

`STEM_MODEL_VERSION` in `packages/core/src/overtone_core/ml/paths.py` must name the version `index` wrote (`v1` on a first build). Without the index, Similar Tracks shows an error and `overtone --self-test` fails. Demucs and MuQ-MuLan download their weights on first use.

To publish a new index version, `python -m overtone_core.ml.model_release pack` writes `dist/models/stem-muq-vN.zip` and prints its sha256. Upload the zip to a GitHub Release tagged `stem-muq-vN`, then move `STEM_MODEL_VERSION` and `STEM_MODEL_SHA256` in `paths.py` together.

The older feature model is kept as the evaluation baseline (downloads an 8 MB public dataset into `data/`, trains into `models/` in about a minute):

```bash
python -m overtone_core.ml.download_dataset
python -m overtone_core.ml.train
```

## Run

```bash
overtone                  # or: python -m overtone. The first run opens a Spotify login in the browser
overtone --self-test      # headless check: scores a catalog song and analyses a few seconds of audio, exits 0 if OK
```

The login token, log, chord cache and the stem vectors of analysed songs live in `%LOCALAPPDATA%\Overtone` (logs in `logs\overtone.log`).

## Development

- `pytest` runs the tests in `tests/`: `unit/` mirrors the app package, `unit/core/` the shared core package and `unit/research/` the research pipeline.
- `pre-commit run --all-files` runs the linters/formatter (ruff) and file hygiene hooks.
- CI (`.github/workflows/ci.yml`) runs both, plus pyright, on every pull request and every push to `main`.
- `python -m overtone_core.ml.evaluate` compares feature-model variants.
- Dependencies are pinned in each package's `pyproject.toml` and locked for the whole workspace in `uv.lock`: after changing one, run `uv lock` and `uv sync`.

### Utility scripts

Run with any Python - each uses the repo's virtualenv itself:

| Script | What it does |
|---|---|
| `scripts/run_checks.py` | What CI runs: pre-commit over all files + the unit tests |
| `scripts/pipeline_status.py` | Research pipeline progress per dataset and stage, what's running, disk, latest log lines |
| `scripts/resume_pipeline.py` | Restarts downloads / GPU stages in the background; skips stages already running |
| `scripts/launch_mlflow.py` | MLflow UI over the research experiment runs (each run has a `report.md`) |
| `scripts/launch_rating_server.py` | The ABX rating app for adding your own similarity judgments |
| `scripts/build_installer.py` | Builds `dist/Overtone-setup-<version>.exe` (needs Inno Setup 6: `winget install JRSoftware.InnoSetup`) |

## Project layout

```
src/overtone/               the app package (python -m overtone)
  config.py                   where the app's files live: assets, app data, caches
  auth.py                     Spotify / OpenAI clients
  assets/                     icon and title font
  ui/                         main window, title bar, list boxes, fullscreen cover, theme
  workers/                    QThreads: playback polling/control, data fetching, similar tracks
  services/                   Spotify, OpenAI chord suggestions (+cache), similar tracks (+stem cache), Win32 window tracking
  music/                      chord-progression schema/validation, MIDI export
packages/core/              overtone_core: shared by the app and the research pipeline
  ml/                         stem model (contract, analysis, scoring), model store, dedup, feature model (baseline)
  clients/                    Deezer and Spotify API clients, HTTP plumbing
  matching.py                 whether a search hit is the recording asked for
  secrets.py, config.py       API keys from .env; repo or install root, HTTP settings
research/stem_similarity/   stem-similarity study (stemsim)
  stemsim/                    datasets, registry, stages (isrc, audio, separate, embed), index, experiments, rating app
scripts/                    standard-library launchers (see above)
installer/                  Inno Setup script for the Windows installer (built by scripts/build_installer.py)
data/                       downloaded dataset (gitignored)
models/                     model versions, models/<model>/<version>/ (gitignored)
dist/                       the built installer and model release zips (gitignored)
tests/                      unit/ (mirrors src/overtone/; unit/core/ and unit/research/ for the other packages) and integration/
uv.lock                     exact versions for the whole workspace
```

## License

All rights reserved - see [LICENSE](LICENSE).

## Contact

For any questions or suggestions, please open an issue or contact me at [nicolas7cunderlik@gmail.com](mailto:nicolas7cunderlik@gmail.com).
