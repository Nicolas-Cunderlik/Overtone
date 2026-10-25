"""Script entry point for PyInstaller (app.spec) - same as `python -m overtone`."""

import sys

from overtone.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
