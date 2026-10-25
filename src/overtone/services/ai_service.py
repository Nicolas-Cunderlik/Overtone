"""AI service: asks the LLM for chord progressions as structured note data
(see music/chord_schema.py) that can be rendered to MIDI.

Flow per track:
1. Cache hit (24h, by track id) -> return it, no API call.
2. Otherwise request the ChordSuggestions schema via Structured Outputs, then
   validate shape + musical consistency. On failure, the model's output and the
   specific problems are sent back and it's asked to fix them - up to
   MAX_RETRIES retries after the first attempt.
3. Only valid output is returned and cached; None if every attempt failed.
"""

import logging

from pydantic import ValidationError

from .. import auth
from ..music.chord_schema import ChordSuggestions, strict_json_schema, validate_suggestions
from .chord_cache import ChordCache

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
REQUEST_TIMEOUT_S = 45
MAX_COMPLETION_TOKENS = 6000

_SYSTEM_PROMPT = """You are a music producer and arranger. Given a song, you write chord \
progressions that capture its vibe - harmony, voicing, rhythm and dynamics - as \
playable note data for a MIDI file.

Rules:
- Write exactly 3 contrasting progressions (e.g. verse-like, chorus-like, a variation).
- Each progression is 2-8 bars long, at a tempo and meter suited to the song.
- Chords are in playing order: the first starts at beat 0 and none overlap.
- Voice each chord with real pitches (MIDI numbers, 60 = middle C), including its \
root or, for slash chords, its bass note. Use idiomatic voicings for the genre \
(e.g. a low bass note plus an open or extended upper structure, power chords for \
rock, 7ths/9ths for soul and jazz) rather than stacked root-position triads.
- Make it sound performed, not quantized - this is what the MIDI is for:
  - Velocity: vary it within every chord and across the progression (bass and \
melody-top notes louder, inner voices softer; build or fall away with the section).
  - Rhythm: chords don't all have to last a full bar; use the song's harmonic \
rhythm, pushes/anticipations and repeated strikes where it suits the style.
  - Offsets: stagger notes a few hundredths of a beat for strums or rolled chords, \
or more for arpeggios - whichever fits the song's instrumentation.
- Titles are short (at most 40 characters) and describe the feel."""

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "chord_suggestions", "strict": True, "schema": strict_json_schema()},
}


class AIService:
    def __init__(self, cache=None):
        self._client = None
        self._cache = cache or ChordCache()

    def _ensure_client(self):
        if self._client is None:
            try:
                self._client = auth.get_openai_client()
            except Exception:
                logger.exception("Failed to initialize OpenAI client")
                self._client = None

    def get_chord_suggestions(self, track_id, track_name, artist):
        """Returns ChordSuggestions, or None if unavailable."""
        if track_id:
            cached = self._cache.get(track_id)
            if cached is not None:
                logger.info("Chord suggestions for %s served from cache", track_id)
                return cached

        self._ensure_client()
        if not self._client:
            return None

        messages = [
            {"role": "developer", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Song: '{track_name}' by {artist}."},
        ]
        for attempt in range(1 + MAX_RETRIES):
            content, problems = self._attempt(messages)
            if not problems:
                suggestions = ChordSuggestions.model_validate_json(content)
                if track_id:
                    self._cache.put(track_id, suggestions)
                if attempt:
                    logger.info("Chord suggestions valid after %d retries", attempt)
                return suggestions

            logger.warning(
                "Chord suggestions attempt %d/%d invalid: %s", attempt + 1, 1 + MAX_RETRIES, "; ".join(problems)
            )
            if content is None:
                continue  # request itself failed - nothing to show the model, just retry
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "That output failed validation:\n- "
                    + "\n- ".join(problems)
                    + "\nReturn the full corrected output.",
                },
            ]

        logger.error(
            "Giving up on chord suggestions for '%s' by %s after %d attempts", track_name, artist, 1 + MAX_RETRIES
        )
        return None

    def _attempt(self, messages):
        """One API call. Returns (raw_content, problems) - problems empty if valid."""
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format=_RESPONSE_FORMAT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception as e:
            logger.exception("AI request failed")
            return None, [f"request failed: {e}"]

        choice = response.choices[0] if response.choices else None
        if choice is None:
            return None, ["empty response"]
        if getattr(choice.message, "refusal", None):
            return None, [f"model refused: {choice.message.refusal}"]
        content = choice.message.content
        if not content:
            return None, ["empty response"]
        if choice.finish_reason == "length":
            # Truncated JSON - feeding it back would only invite another cut-off
            return None, ["output was truncated (hit the token limit)"]

        try:
            suggestions = ChordSuggestions.model_validate_json(content)
        except ValidationError as e:
            return content, [
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" if err['loc'] else err['msg'] for err in e.errors()
            ]
        except ValueError as e:
            return content, [f"not valid JSON: {e}"]
        return content, validate_suggestions(suggestions)
