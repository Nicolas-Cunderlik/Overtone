import pytest

from overtone.music import midi


@pytest.mark.parametrize(
    "value, encoded",
    [
        # Examples from the Standard MIDI File spec
        (0x00, b"\x00"),
        (0x7F, b"\x7f"),
        (0x80, b"\x81\x00"),
        (0x3FFF, b"\xff\x7f"),
        (0x0FFFFFFF, b"\xff\xff\xff\x7f"),
    ],
)
def test_variable_length_quantity(value, encoded):
    assert midi._vlq(value) == encoded


@pytest.mark.parametrize(
    "key, expected",
    [
        ("C major", (0, False)),
        ("F# minor", (3, True)),
        ("Bb major", (-2, False)),
        ("Am", (0, True)),
        ("not a key", None),
    ],
)
def test_key_signature(key, expected):
    assert midi._key_signature(key) == expected
