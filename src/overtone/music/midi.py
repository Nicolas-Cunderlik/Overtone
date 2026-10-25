"""Converts a validated `Progression` into a Standard MIDI File (format 0).

Written by hand rather than with a MIDI library - the subset needed (one
track: name, tempo, time/key signature, piano program, note on/off) is small,
and it keeps the app free of another dependency.

Output: one track on channel 1 with an acoustic piano program, so it plays
back immediately in any DAW or media player and can be re-voiced from there.
"""

import re
import struct

TICKS_PER_BEAT = 480
CHANNEL = 0
PIANO_PROGRAM = 0

# Key signature as (sharps(+)/flats(-), is_minor), for the key-signature meta
# event. Spelled as the LLM tends to write keys ("F# minor", "Bb major").
_MAJOR_SHARPS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
}
_MINOR_SHARPS = {
    "A": 0,
    "E": 1,
    "B": 2,
    "F#": 3,
    "C#": 4,
    "G#": 5,
    "D#": 6,
    "A#": 7,
    "D": -1,
    "G": -2,
    "C": -3,
    "F": -4,
    "Bb": -5,
    "Eb": -6,
    "Ab": -7,
}


def _vlq(value):
    """MIDI variable-length quantity."""
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _meta(meta_type, data):
    return b"\xff" + bytes([meta_type]) + _vlq(len(data)) + data


def _key_signature(key):
    """Parses e.g. 'F# minor' / 'Bb major' / 'Am'; None if unrecognised."""
    match = re.match(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(m(?:in(?:or)?)?|maj(?:or)?)?\b", key or "", re.IGNORECASE)
    if not match:
        return None
    tonic = match.group(1).upper() + match.group(2).replace("♯", "#").replace("♭", "b")
    quality = (match.group(3) or "major").lower()
    minor = quality.startswith("m") and not quality.startswith("maj")
    table = _MINOR_SHARPS if minor else _MAJOR_SHARPS
    if tonic not in table:
        return None
    return table[tonic], minor


def _ticks(beats):
    return round(beats * TICKS_PER_BEAT)


def progression_to_midi(progression):
    """Returns the .mid file contents as bytes."""
    header_events = [
        (0, _meta(0x03, progression.title.encode("utf-8"))),  # track name
        (0, _meta(0x51, struct.pack(">I", round(60_000_000 / progression.tempo_bpm))[1:])),  # tempo, µs per beat
        (0, _meta(0x58, bytes([progression.beats_per_bar, 2, 24, 8]))),  # time signature, /4
        (0, bytes([0xC0 | CHANNEL, PIANO_PROGRAM])),  # program change
    ]
    key = _key_signature(progression.key)
    if key is not None:
        sharps, minor = key
        header_events.append((0, _meta(0x59, struct.pack(">bB", sharps, int(minor)))))

    # (tick, order, bytes). Note-offs sort before note-ons at the same tick so a
    # pitch repeated across consecutive chords re-triggers instead of being cut.
    note_events = []
    for chord in progression.chords:
        for note in chord.notes:
            start = _ticks(chord.start_beat + note.offset_beats)
            end = max(start + 1, _ticks(chord.start_beat + note.offset_beats + note.duration_beats))
            note_events.append((start, 1, bytes([0x90 | CHANNEL, note.pitch, note.velocity])))
            note_events.append((end, 0, bytes([0x80 | CHANNEL, note.pitch, 0])))
    note_events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    now = 0
    for tick, data in header_events + [(t, d) for t, _o, d in note_events]:
        track += _vlq(tick - now) + data
        now = tick
    track += _vlq(0) + _meta(0x2F, b"")  # end of track

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, TICKS_PER_BEAT)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
