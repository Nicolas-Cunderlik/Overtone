import pytest
from overtone_core.matching import same_recording


@pytest.mark.parametrize(
    ("wanted", "found", "durations_match", "expected"),
    [
        ("Wake Me Up - Radio Edit", "Wake Me Up", False, True),  # packaging words dropped
        ("Girl I'm on the Way", "Girl Im on the Way", False, True),  # typo
        ("C.R.E.A.M.", "C.R.E.A.M. (Cash Rules Everything Around Me)", True, True),
        ("C.R.E.A.M.", "C.R.E.A.M. (Cash Rules Everything Around Me)", False, False),
        ("Close To Me", "Close To Me (Live)", True, False),
        ("Blue Sky - Michael Brun Remix", "Blue Sky (Michael Brun Extended Remix)", True, False),
        ("Skyline - Klave Remix", "Skyline (Alex Schulz Remix)", True, False),
    ],
)
def test_same_recording(wanted, found, durations_match, expected):
    assert same_recording(wanted, found, durations_match=durations_match) is expected
