# Third-party notices

Overtone itself is proprietary (see LICENSE). The components below keep their
own licenses. The installer does not bundle them: it downloads each one from
its publisher (PyPI, the PyTorch package index, Hugging Face, GitHub) during
setup.

## Models

- **MuQ-MuLan** (`OpenMuQ/MuQ-MuLan-large`), Tencent AI Lab.
  The model weights are licensed under Creative Commons
  Attribution-NonCommercial 4.0 (CC BY-NC 4.0):
  https://creativecommons.org/licenses/by-nc/4.0/
  **The weights may only be used for non-commercial purposes.**
  Code: https://github.com/tencent-ailab/MuQ (MIT License).
- **XLM-RoBERTa** (`xlm-roberta-base`), Meta AI, downloaded by MuQ-MuLan for
  its text encoder. MIT License.
- **Demucs** (`htdemucs`), Meta AI. Code and pretrained weights are under the
  MIT License: https://github.com/adefossez/demucs

## Catalog index

The catalog index (`models/stem-muq/`) holds embeddings that Overtone computed
from 30-second preview clips served by the Deezer API. It contains no audio:
no preview clips or separated stems are distributed.

## Software

Installed into Overtone's own Python environment, each under its own license
(the license files are in `.venv\Lib\site-packages\*.dist-info`):

- Python, Python Software Foundation License
- PyTorch, BSD-3-Clause and other permissive licenses; torchaudio, BSD
- PyQt5, GNU GPL v3 (Riverbank Computing)
- NumPy, pandas, scikit-learn, BSD-3-Clause
- Hugging Face Transformers, Apache License 2.0
- spotipy, MIT; OpenAI Python library, Apache License 2.0
- uv (used by the installer), Apache License 2.0 or MIT
