"""Output schema for LLM-generated chord progressions.

Two layers of validation:

1. Shape and ranges - the pydantic models below. `strict_json_schema()` turns
   them into the JSON schema sent to the model as a Structured Outputs
   response format, so the API itself enforces the shape.
2. Musical consistency - `validate_suggestions()`. Rules a JSON schema can't
   express: the notes actually contain the named chord's root (or slash bass),
   chords run in order without overlapping, notes don't spill far past their
   chord. Its messages are written to be fed back to the model on a retry.

Timing is in beats (quarter notes) so it maps directly onto MIDI ticks.
"""

import copy
import re

from pydantic import BaseModel, Field

MAX_PROGRESSION_BEATS = 64  # 16 bars of 4/4
MAX_NOTE_OVERHANG_BEATS = 4  # how far a note may ring past its chord's end
MAX_TITLE_LENGTH = 40


class Note(BaseModel):
    pitch: int = Field(
        ge=21, le=108, description="MIDI note number, 60 = middle C. Keep voicings roughly within 36-96."
    )
    velocity: int = Field(ge=1, le=127, description="MIDI velocity; vary it for dynamics and accents.")
    offset_beats: float = Field(
        ge=0,
        le=16,
        description="Start relative to the chord's start_beat. 0 = struck with the chord; "
        "small staggered offsets for strums or arpeggios.",
    )
    duration_beats: float = Field(gt=0, le=16, description="How long the note sounds, in beats.")


class Chord(BaseModel):
    symbol: str = Field(description='Chord symbol, e.g. "Cmaj7", "F#m", "Bb7sus4", "G/B".')
    start_beat: float = Field(ge=0, description="Start of the chord in beats from the start of the progression.")
    duration_beats: float = Field(gt=0, le=16, description="Length of the chord in beats.")
    notes: list[Note] = Field(min_length=1, max_length=8, description="The voicing - the actual pitches played.")


class Progression(BaseModel):
    title: str = Field(description=f"Short description of the feel, at most {MAX_TITLE_LENGTH} characters.")
    key: str = Field(description='Key of the progression, e.g. "F minor", "D major".')
    tempo_bpm: int = Field(ge=40, le=220)
    beats_per_bar: int = Field(
        ge=2, le=7, description="Beats per bar; the beat is a quarter note (e.g. 4 for 4/4, 3 for 3/4)."
    )
    chords: list[Chord] = Field(min_length=2, max_length=16, description="In playing order, starting at beat 0.")


class ChordSuggestions(BaseModel):
    progressions: list[Progression] = Field(min_length=3, max_length=3)


def strict_json_schema(model=ChordSuggestions):
    """The model's JSON schema in the form OpenAI's strict Structured Outputs
    mode requires: every object closed (additionalProperties false) and every
    property listed as required. All fields above are required anyway."""
    schema = copy.deepcopy(model.model_json_schema())

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    return schema


_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ROOT = re.compile(r"^\s*([A-G])([#b♯♭]?)")
_SLASH_BASS = re.compile(r"/\s*([A-G])([#b♯♭]?)\s*$")


def _pitch_class(letter, accidental):
    shift = {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(accidental, 0)
    return (_NOTE_NAMES[letter] + shift) % 12


def validate_suggestions(suggestions):
    """Returns a list of human-readable problems (empty if the output is usable)."""
    problems = []
    for p_index, progression in enumerate(suggestions.progressions, start=1):
        where = f"progression {p_index}"
        if not progression.title.strip():
            problems.append(f"{where}: title is empty")
        elif len(progression.title) > MAX_TITLE_LENGTH:
            problems.append(f"{where}: title is {len(progression.title)} characters, max {MAX_TITLE_LENGTH}")

        previous_end = 0.0
        for c_index, chord in enumerate(progression.chords, start=1):
            at = f"{where}, chord {c_index} ({chord.symbol!r})"
            root = _ROOT.match(chord.symbol)
            if not root:
                problems.append(f"{at}: symbol must start with a root note A-G")
            else:
                pitch_classes = {note.pitch % 12 for note in chord.notes}
                expected = {_pitch_class(*root.groups())}
                bass = _SLASH_BASS.search(chord.symbol)
                if bass:
                    expected.add(_pitch_class(*bass.groups()))
                if not expected & pitch_classes:
                    problems.append(f"{at}: voicing contains neither the chord's root nor its bass note")

            if c_index == 1 and chord.start_beat != 0:
                problems.append(f"{at}: the first chord must start at beat 0")
            if chord.start_beat < previous_end - 1e-6:
                problems.append(
                    f"{at}: starts at beat {chord.start_beat}, "
                    f"overlapping the previous chord which ends at {previous_end}"
                )
            previous_end = chord.start_beat + chord.duration_beats

            pitches = [note.pitch for note in chord.notes]
            if len(pitches) != len(set(pitches)):
                problems.append(f"{at}: the same pitch appears twice in one voicing")
            for note in chord.notes:
                if note.offset_beats >= chord.duration_beats:
                    problems.append(f"{at}: a note starts at offset {note.offset_beats}, after the chord has ended")
                if note.offset_beats + note.duration_beats > chord.duration_beats + MAX_NOTE_OVERHANG_BEATS:
                    problems.append(
                        f"{at}: a note rings more than {MAX_NOTE_OVERHANG_BEATS} beats past the chord's end"
                    )

        if previous_end > MAX_PROGRESSION_BEATS:
            problems.append(f"{where}: lasts {previous_end} beats, max {MAX_PROGRESSION_BEATS}")
    return problems
