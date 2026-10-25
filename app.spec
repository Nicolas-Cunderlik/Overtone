# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build for Overtone: a single windowed exe.
#   pip install -e ".[dev]"
#   python -m overtone.ml.train      (models/ is bundled, so train first)
#   pyinstaller app.spec               -> dist/Overtone.exe
#
# Secrets are NOT bundled: put the .env next to the exe (or point the
# OVERTONE_ENV_FILE env var at it). The login token, log and chord cache go
# to %LOCALAPPDATA%\Overtone. See src/overtone/config.py.
import os

# Bundle paths mirror the source layout, so config.PROJECT_ROOT /
# config.ASSET_DIR resolve the same way inside the exe (sys._MEIPASS)
if not os.path.isdir("models/similarity"):
    raise SystemExit("Model artifacts missing - run `python -m overtone.ml.train` first")
datas = [
    ("src/overtone/assets", "overtone/assets"),  # the title font is optional (gitignored)
    ("models/similarity", "models/similarity"),
]

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Overtone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
