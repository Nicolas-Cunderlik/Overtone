"""The stem model's audio contract: which channels it embeds, in what order,
at what rate, with which model. The research pipeline (which builds the
catalog index), the index loader and live analysis all follow it, so a
catalog song and a live song are comparable.

Torch-free, so code that only needs the names (the rating app, scoring)
doesn't import torch. stem_analysis.py holds the code that does the work.
"""

STEMS = ("drums", "bass", "vocals", "other")  # Demucs htdemucs stems
MIX = "mix"
# Embedding channels; fitted stem weights are positional, so every user takes this order
CHANNELS = (MIX, *STEMS)

DEMUCS_MODEL = "htdemucs"  # separates the mix into STEMS
STEM_SR = 24_000  # MuQ's input rate; stems are resampled to it
EMBED_MODEL = "OpenMuQ/MuQ-MuLan-large"  # Hugging Face id
EMBED_DIMS = 512
MIN_WINDOW_S = 1.0  # shorter audio (e.g. Deezer's ~1 s stub instead of a preview) gives no usable embedding
