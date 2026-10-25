import os
import tempfile

# Runs before any test imports app code: overtone.config resolves (and, on a
# machine that still has the pre-rename Spotistats dir, migrates) the per-user
# data dir at import. Tests must never touch the real one.
os.environ.setdefault("OVERTONE_DATA_DIR", tempfile.mkdtemp(prefix="overtone-test-"))
