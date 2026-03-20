"""
core/voice/speech_to_text.py
─────────────────────────────
Whisper-based speech-to-text that can:
  • Transcribe an audio *file* (WAV / MP3 / M4A …).
  • Record from the microphone and transcribe on-the-fly.

The Whisper model is lazy-loaded once and cached.
"""
from __future__ import annotations

import io
import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# ─── Whisper lazy load ───────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import whisper  # type: ignore[import]

                logger.info("Loading Whisper model: %s", settings.whisper_model)
                _whisper_model = whisper.load_model(settings.whisper_model)
    return _whisper_model


# ─── Public helpers ──────────────────────────────────────────────────────────

def transcribe_file(audio_path: str | Path) -> str:
    """
    Transcribe an audio file and return the text.

    Args:
        audio_path: Path to a WAV / MP3 / M4A / FLAC file.

    Returns:
        Transcribed text string.
    """
    model = _get_whisper()
    result = model.transcribe(
        str(audio_path),
        language=settings.voice_language,
        fp16=False,
    )
    text: str = result["text"].strip()
    logger.debug("Transcribed file '%s' → %s", audio_path, text[:80])
    return text


def transcribe_numpy(audio_array: np.ndarray, sample_rate: int = 16_000) -> str:
    """
    Transcribe a NumPy float32 audio array (mono, 16 kHz expected).

    Whisper internally resamples if needed, but 16 kHz is most efficient.
    """
    import whisper  # type: ignore[import]

    model = _get_whisper()
    # Whisper expects float32 mono; normalise if necessary
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Pad/trim to Whisper's expected 30-second chunk
    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(model.device)
    options = whisper.DecodingOptions(
        language=settings.voice_language, fp16=False
    )
    result = whisper.decode(model, mel, options)
    text: str = result.text.strip()
    logger.debug("Transcribed numpy array → %s", text[:80])
    return text


class MicrophoneListener:
    """
    Continuously records from the default microphone.
    Call ``listen()`` to block until a phrase is captured,
    then returns its transcription.

    Uses sounddevice for capture and a simple energy-based VAD
    (Voice Activity Detection) to detect speech boundaries.
    """

    SAMPLE_RATE = 16_000
    BLOCK_DURATION = 0.03          # 30 ms blocks
    SILENCE_THRESHOLD = 0.01       # RMS level below which it's silence
    SILENCE_DURATION = 1.5         # seconds of silence to stop recording
    MIN_SPEECH_DURATION = 0.5      # minimum speech seconds to keep

    def __init__(self) -> None:
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "sounddevice is not installed. Run: pip install sounddevice"
            ) from exc

    # ── Internal helpers ────────────────────────────────────────────────────

    def _rms(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block**2)))

    # ── Public API ──────────────────────────────────────────────────────────

    def listen(self, timeout: float = 30.0) -> Optional[str]:
        """
        Block until the user speaks and finishes (silence detected).
        Returns the transcription, or ``None`` on timeout.
        """
        import sounddevice as sd

        block_size = int(self.SAMPLE_RATE * self.BLOCK_DURATION)
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        stop_event = threading.Event()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if not stop_event.is_set():
                audio_queue.put(indata.copy())

        frames_collected: list[np.ndarray] = []
        silence_frames = 0
        max_silence_frames = int(self.SILENCE_DURATION / self.BLOCK_DURATION)
        speech_started = False

        logger.info("🎙  Listening… (speak now)")

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=_callback,
        ):
            total_frames = 0
            max_frames = int(timeout / self.BLOCK_DURATION)

            while total_frames < max_frames:
                try:
                    block = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                rms = self._rms(block)
                total_frames += 1

                if rms > self.SILENCE_THRESHOLD:
                    speech_started = True
                    silence_frames = 0
                    frames_collected.append(block)
                elif speech_started:
                    frames_collected.append(block)
                    silence_frames += 1
                    if silence_frames >= max_silence_frames:
                        break  # enough silence → end of utterance

            stop_event.set()

        if not frames_collected:
            logger.warning("No speech detected within timeout.")
            return None

        audio = np.concatenate(frames_collected, axis=0).flatten()
        speech_duration = len(audio) / self.SAMPLE_RATE

        if speech_duration < self.MIN_SPEECH_DURATION:
            logger.warning("Recording too short (%.2fs), ignoring.", speech_duration)
            return None

        logger.info("🔇  Recording stopped (%.2fs of audio).", speech_duration)
        return transcribe_numpy(audio, self.SAMPLE_RATE)
