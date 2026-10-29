"""One-screen pipeline overview: per-dataset stage progress, which stages are
running right now, disk use, and the latest line of each stage log.

    python scripts/pipeline_status.py
"""

import shutil

from _common import LOG_DIR, RESEARCH, run, running_stages, venv_python


def last_progress_line(path):
    """The last INFO line of a log (skipping library noise like decoder warnings)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return next((line for line in reversed(lines) if " INFO " in line), None)


def main():
    code = run([venv_python(), "-m", "stemsim", "status"], cwd=RESEARCH)

    held = running_stages()
    print("\nRunning now:", ", ".join(held) if held else "nothing")

    total, _used, free = shutil.disk_usage(RESEARCH)
    print(f"Disk: {free / 1e9:.0f} GB free of {total / 1e9:.0f} GB")

    if LOG_DIR.exists():
        print("\nLatest log lines:")
        for log in sorted(LOG_DIR.glob("*.log")):
            line = last_progress_line(log)
            if line:
                print(f"  {log.stem:16s} {line[:130]}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
