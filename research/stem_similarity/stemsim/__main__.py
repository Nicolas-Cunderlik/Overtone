"""Command line for the pipeline. Run from research/stem_similarity with its venv:

  python -m stemsim datasets                         list adapters
  python -m stemsim ingest spotify_catalog           register the app's catalog (seeds)
  python -m stemsim ingest magnatagatune [--with-audio]
  python -m stemsim ingest ccmsim [--all-cultures]
  python -m stemsim ingest lastfm [--limit 500]      crawl 500 more seeds, then re-normalize
  python -m stemsim isrc [--dataset dim_sim] [--limit 1000]  Spotify ISRCs (Deezer audio waits for these)
  python -m stemsim audio [--source deezer] [--dataset dim_sim] [--limit 1000]
                         isrc and audio skip Last.fm-only tracks unless given --dataset lastfm
  python -m stemsim separate [--dataset dim_sim] [--limit 100]   Demucs stems (GPU)
  python -m stemsim embed [--dataset dim_sim] [--limit 100]      MuQ-MuLan vectors (GPU)
  python -m stemsim status                           per-dataset progress
  python -m stemsim session [--rater nicolas]        build the next rating session (rating/sessions.py)
  python -m stemsim rate [--session NAME]            serve a session to rate (rating/server.py)
  python -m stemsim index --weights-run RUN_ID       next stem-muq model version (index.py)

Every command is safe to re-run and to run alongside the others; a second
copy of the same stage refuses to start. GPU stages without --dataset work
through datasets in PRIORITY order, so evaluation data is ready first.
"""

import argparse
import contextlib
import logging
import sys

from . import config
from .datasets import ADAPTERS
from .registry import Registry
from .stages import audio, ingest, isrc

# Perceptual targets first (they gate training/evaluation), the candidate pool last
PRIORITY = ("listener_abx", "magnatagatune", "ccmsim", "dim_sim", "matcha", "spotify_catalog", "lastfm")

_last_reported = [0]


def _progress(done, total):
    """Prints download progress each time another 256 MB has arrived."""
    if done < _last_reported[0]:  # a new file started
        _last_reported[0] = 0
    if done - _last_reported[0] >= 256 << 20 or done == total:
        _last_reported[0] = done
        print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)


def _run_by_priority(registry, stage_run, dataset, limit):
    """Runs a stage on one dataset, or on every dataset in PRIORITY order."""
    totals = {}
    for name in [dataset] if dataset else PRIORITY:
        remaining = None if limit is None else limit - sum(totals.values())
        if remaining is not None and remaining <= 0:
            break
        for state, n in stage_run(registry, dataset=name, limit=remaining).items():
            totals[state] = totals.get(state, 0) + n
        logging.getLogger(__name__).info("%s: %s", name, totals)
    return totals


def _weights_from_run(run_id, registry):
    """index.build()'s weights, silent_db and source, from a stem-weights MLflow run."""
    import json

    import mlflow  # not installed in the app venv the unit tests run in

    from . import tracking

    mlflow.set_tracking_uri(tracking.tracking_uri())
    run = mlflow.get_run(run_id)
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights.json")
    with open(path, encoding="utf-8") as f:
        weights = json.load(f)
    source = {
        "weights_run": run_id,
        "weights_run_name": run.info.run_name,
        "weights_git_commit": run.data.tags.get("git_commit"),
        "data": {
            name: embedded
            for name, (_judgments, embedded) in tracking.data_snapshot(registry, config.EMBED_STAGE).items()
        },
    }
    return {"weights": weights, "silent_db": float(run.data.params["silent_db"]), "source": source}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stemsim", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("datasets", help="list dataset adapters")
    p = commands.add_parser("ingest", help="fetch + normalize a dataset into the registry")
    p.add_argument("dataset", choices=sorted(ADAPTERS))
    p.add_argument("--no-fetch", action="store_true", help="re-normalize cached raw files only")
    p.add_argument("--with-audio", action="store_true", help="magnatagatune: also download the 3.3 GB audio")
    p.add_argument("--all-cultures", action="store_true", help="ccmsim: keep all nine source datasets")
    p.add_argument("--limit", type=int, default=200, help="lastfm: seeds to crawl this run")
    no_audio = ", ".join(config.NO_AUDIO_DATASETS)
    dataset_help = f"only this dataset's tracks; without it, tracks that only {no_audio} includes are skipped"
    p = commands.add_parser("isrc", help="Spotify ISRCs for Deezer tracks still without audio")
    p.add_argument("--dataset", choices=sorted(ADAPTERS), help=dataset_help)
    p.add_argument("--limit", type=int)
    p = commands.add_parser("audio", help="fetch/extract audio for pending tracks")
    p.add_argument("--source")
    p.add_argument("--dataset", help=dataset_help)
    p.add_argument("--limit", type=int)
    for name, help_text in (("separate", "Demucs stems for tracks with audio"), ("embed", "MuQ-MuLan vectors")):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--dataset", choices=sorted(ADAPTERS))
        p.add_argument("--limit", type=int)
    commands.add_parser("status", help="per-dataset progress")
    p = commands.add_parser("session", help="build the next ABX rating session")
    p.add_argument("--rater", default=config.DEFAULT_RATER)
    p.add_argument("--seed", type=int, help="for a reproducible session")
    p.add_argument("--dry-run", action="store_true", help="print the session without writing it")
    commands.add_parser("rate", help="serve a session in the ABX rating app (rate --help for its options)")
    p = commands.add_parser("index", help="build the next stem-muq model version from the catalog's embeddings")
    p.add_argument("--weights-run", required=True, help="stem-weights MLflow run whose fitted weights to ship")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["rate"]:  # the server parses its own options and needs no registry
        from .rating import server

        return server.main(argv[1:])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    registry = Registry()

    if args.command == "datasets":
        for name, adapter in ADAPTERS.items():
            print(f"{name:16s} {adapter.target:11s} {adapter.metric or '-':18s} {adapter.url}")
    elif args.command == "ingest":
        options = {"cultures": None} if args.all_cultures else {}
        with registry.exclusive(f"ingest-{args.dataset}"):
            ingest.ingest(
                registry,
                args.dataset,
                fetch=not args.no_fetch,
                adapter_options=options,
                with_audio=args.with_audio,
                limit=args.limit,
                progress=_progress,
            )
    elif args.command == "isrc":
        with registry.exclusive(isrc.STAGE):
            print(isrc.run(registry, dataset=args.dataset, limit=args.limit))
    elif args.command == "audio":
        # One process per audio source (they'd write the same files); without
        # --source this run touches every source, so it holds all their locks
        sources = [args.source] if args.source else [*audio.RESOLVERS, "other"]
        with contextlib.ExitStack() as locks:
            for source in sources:
                locks.enter_context(registry.exclusive(f"audio-{source}"))
            print(audio.run(registry, source=args.source, dataset=args.dataset, limit=args.limit))
    elif args.command in ("separate", "embed"):
        # torch/demucs/muq are imported only here - the rest of the CLI runs without them
        if args.command == "separate":
            from .stages import separate as stage
        else:
            from .stages import embed as stage
        with registry.exclusive(stage.STAGE.replace(":", "-")):
            print(_run_by_priority(registry, stage.run, args.dataset, args.limit))
    elif args.command == "session":
        from .rating import sessions  # loads scipy, mlflow and the app model

        name, rows = sessions.build(registry, args.rater, seed=args.seed)
        print(sessions.summary(rows))
        if args.dry_run:
            print(f"(dry run - {name} not written)")
        else:
            print(f"Wrote {sessions.write(rows, name)} - rate it with scripts/launch_rating_server.py")
    elif args.command == "index":
        from . import index

        built = index.build(registry, **_weights_from_run(args.weights_run, registry))
        print(f"Wrote {built.path}")
    elif args.command == "status":
        stages = [config.ISRC_STAGE, config.AUDIO_STAGE, config.SEPARATE_STAGE, config.EMBED_STAGE]
        print(
            f"{'dataset':16s} {'target':11s} {'tracks':>7s} {'judgm.':>7s} "
            + " ".join(f"{s[:12]:>18s}" for s in stages)
        )
        print(f"{'':45s}" + " ".join(f"{'ok / fail / n.a.':>18s}" for _ in stages))
        for row, by_stage in registry.summary(stages):
            cells = " ".join(f"{c['done']:>8d}/{c['failed']:>4d}/{c['unavailable']:>4d}" for c in by_stage.values())
            print(f"{row['name']:16s} {row['target']:11s} {row['n_tracks']:7d} {row['n_judgments']:7d} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
