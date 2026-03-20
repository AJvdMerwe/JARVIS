"""
core/voice/text_to_speech.py
─────────────────────────────
Offline TTS using pyttsx3.
Provides a simple ``speak(text)`` function and a streaming-friendly
``SpeechSynthesiser`` class.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    import pyttsx3  # type: ignore[import]
                    _engine = pyttsx3.init()
                    _engine.setProperty("rate", 175)   # words per minute
                    _engine.setProperty("volume", 0.9)
                    logger.info("TTS engine initialised (pyttsx3).")
                except Exception as exc:
                    logger.warning("pyttsx3 init failed: %s – voice output disabled.", exc)
    return _engine


def speak(text: str, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
    """
    Speak *text* synchronously using the system TTS engine.

    Args:
        text:   The text to synthesise.
        rate:   Speech rate in words/min (default 175).
        volume: Volume 0.0–1.0 (default 0.9).
    """
    engine = _get_engine()
    if engine is None:
        logger.warning("TTS unavailable; would have said: %s", text[:80])
        return

    with _engine_lock:
        if rate:
            engine.setProperty("rate", rate)
        if volume:
            engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()


class SpeechSynthesiser:
    """
    Thread-safe TTS helper.
    Useful when responses are streamed token-by-token: call ``feed()`` to
    accumulate sentence fragments and ``flush()`` to speak completed sentences
    as they arrive.
    """

    _SENTENCE_ENDINGS = {".", "!", "?", "…"}

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> None:
        """Append a streamed token and speak any completed sentence immediately."""
        self._buffer += token
        # Look for a sentence boundary
        for i, char in enumerate(self._buffer):
            if char in self._SENTENCE_ENDINGS:
                sentence = self._buffer[: i + 1].strip()
                self._buffer = self._buffer[i + 1 :]
                if sentence:
                    speak(sentence)
                return

    def flush(self) -> None:
        """Speak any remaining buffered text."""
        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder:
            speak(remainder)
