"""
tests/test_voice.py
────────────────────
Unit tests for voice components (Whisper STT, pyttsx3 TTS).
All external dependencies (whisper, sounddevice, pyttsx3) are mocked.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
#  Speech-to-text
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscribeFile:
    def test_transcribe_calls_whisper(self, tmp_path: Path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello world  "}

        with (
            patch("core.voice.speech_to_text._whisper_model", mock_model),
            patch("core.voice.speech_to_text._get_whisper", return_value=mock_model),
        ):
            from core.voice.speech_to_text import transcribe_file

            audio_file = tmp_path / "audio.wav"
            audio_file.write_bytes(b"fake audio")
            result = transcribe_file(audio_file)

        assert result == "Hello world"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_strips_whitespace(self, tmp_path: Path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "   leading and trailing   "}

        with patch("core.voice.speech_to_text._get_whisper", return_value=mock_model):
            from core.voice.speech_to_text import transcribe_file

            audio_file = tmp_path / "audio.wav"
            audio_file.write_bytes(b"fake")
            result = transcribe_file(audio_file)

        assert result == "leading and trailing"


class TestTranscribeNumpy:
    def test_transcribe_numpy_float32(self):
        mock_whisper_mod = MagicMock()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "  Test transcription  "

        mock_whisper_mod.pad_or_trim.side_effect = lambda x: x
        mock_whisper_mod.log_mel_spectrogram.return_value = MagicMock(to=MagicMock(return_value=MagicMock()))
        mock_whisper_mod.DecodingOptions.return_value = MagicMock()
        mock_whisper_mod.decode.return_value = mock_result
        mock_model.device = "cpu"

        audio = np.zeros(16000, dtype=np.float32)

        with (
            patch("core.voice.speech_to_text._get_whisper", return_value=mock_model),
            patch("core.voice.speech_to_text.whisper", mock_whisper_mod),
        ):
            from core.voice.speech_to_text import transcribe_numpy
            import importlib, core.voice.speech_to_text as stt_mod
            stt_mod.whisper = mock_whisper_mod

            # Patch the whisper import inside the function
            with patch.dict("sys.modules", {"whisper": mock_whisper_mod}):
                mock_whisper_mod.pad_or_trim.return_value = audio
                result = transcribe_numpy(audio)

        # Just verify it runs without error (whisper internals are mocked)
        assert isinstance(result, str)

    def test_int16_audio_converted(self):
        """int16 audio should be cast to float32 without error."""
        audio_int16 = (np.random.randn(16000) * 32767).astype(np.int16)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_result = MagicMock()
        mock_result.text = "ok"

        mock_whisper = MagicMock()
        mock_whisper.pad_or_trim.side_effect = lambda x: x
        mock_whisper.log_mel_spectrogram.return_value = MagicMock(to=MagicMock(return_value=MagicMock()))
        mock_whisper.DecodingOptions.return_value = MagicMock()
        mock_whisper.decode.return_value = mock_result

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            import core.voice.speech_to_text as stt_mod
            stt_mod.whisper = mock_whisper

            with patch("core.voice.speech_to_text._get_whisper", return_value=mock_model):
                result = stt_mod.transcribe_numpy(audio_int16)

        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
#  Text-to-speech
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeak:
    def test_speak_calls_engine(self):
        mock_engine = MagicMock()

        with (
            patch("core.voice.text_to_speech._get_engine", return_value=mock_engine),
            patch("core.voice.text_to_speech._engine_lock"),
        ):
            from core.voice.text_to_speech import speak
            speak("Hello, this is a test.")

        mock_engine.say.assert_called_once_with("Hello, this is a test.")
        mock_engine.runAndWait.assert_called_once()

    def test_speak_with_none_engine(self):
        """Should not raise when engine is unavailable."""
        with patch("core.voice.text_to_speech._get_engine", return_value=None):
            from core.voice.text_to_speech import speak
            speak("Should not crash")  # Must not raise


class TestSpeechSynthesiser:
    def test_feed_speaks_at_sentence_boundary(self):
        from core.voice.text_to_speech import SpeechSynthesiser

        synth = SpeechSynthesiser()
        spoken: list[str] = []

        with patch("core.voice.text_to_speech.speak", side_effect=spoken.append):
            synth.feed("This is a sentence")
            assert len(spoken) == 0
            synth.feed(".")
            assert len(spoken) == 1
            assert "This is a sentence." in spoken[0]

    def test_flush_speaks_remainder(self):
        from core.voice.text_to_speech import SpeechSynthesiser

        synth = SpeechSynthesiser()
        synth._buffer = "Unfinished fragment"
        spoken: list[str] = []

        with patch("core.voice.text_to_speech.speak", side_effect=spoken.append):
            synth.flush()

        assert len(spoken) == 1
        assert "Unfinished fragment" in spoken[0]

    def test_flush_empty_buffer(self):
        from core.voice.text_to_speech import SpeechSynthesiser

        synth = SpeechSynthesiser()
        synth._buffer = ""
        with patch("core.voice.text_to_speech.speak") as mock_speak:
            synth.flush()
        mock_speak.assert_not_called()

    def test_multiple_sentences(self):
        from core.voice.text_to_speech import SpeechSynthesiser

        synth = SpeechSynthesiser()
        spoken: list[str] = []

        with patch("core.voice.text_to_speech.speak", side_effect=spoken.append):
            for char in "First sentence. Second one!":
                synth.feed(char)
            synth.flush()

        assert len(spoken) >= 2


# ─────────────────────────────────────────────────────────────────────────────
#  MicrophoneListener (structure test only — no real microphone)
# ─────────────────────────────────────────────────────────────────────────────

class TestMicrophoneListener:
    def test_instantiation_requires_sounddevice(self):
        """Should raise ImportError if sounddevice is not installed."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            import importlib
            import core.voice.speech_to_text as stt_mod
            importlib.reload(stt_mod)

            with pytest.raises((ImportError, TypeError)):
                stt_mod.MicrophoneListener()

    def test_rms_calculation(self):
        from core.voice.speech_to_text import MicrophoneListener
        with patch("sounddevice.InputStream"):
            listener = MicrophoneListener.__new__(MicrophoneListener)

        # Silent signal
        silent = np.zeros(1000, dtype=np.float32)
        assert listener._rms(silent) == 0.0

        # Non-silent signal
        loud = np.ones(1000, dtype=np.float32)
        assert listener._rms(loud) == pytest.approx(1.0, rel=1e-3)
