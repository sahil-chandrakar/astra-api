import asyncio
import tempfile
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import VoiceTranscriptionResponse


class SpeechToTextService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None
        self._model_name = ""

    async def transcribe_upload(self, filename: str, content: bytes) -> VoiceTranscriptionResponse:
        if not content:
            return VoiceTranscriptionResponse(transcript="", message="No audio was received.", setup_required=[])
        if len(content) > self.settings.stt_max_upload_bytes:
            return VoiceTranscriptionResponse(
                transcript="",
                message="Voice recording is too large. Try a shorter command.",
                setup_required=[],
            )

        suffix = self._safe_suffix(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        try:
            transcript = await asyncio.to_thread(self._transcribe_file, temp_path)
        except ImportError:
            return VoiceTranscriptionResponse(
                transcript="",
                message="Local voice input needs faster-whisper installed.",
                setup_required=["faster-whisper"],
            )
        except Exception as exc:
            return VoiceTranscriptionResponse(
                transcript="",
                message=f"Local voice transcription failed: {exc}",
                setup_required=[],
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if not transcript:
            return VoiceTranscriptionResponse(transcript="", message="Listening locally. No speech detected.", setup_required=[])
        return VoiceTranscriptionResponse(transcript=transcript, message="Transcribed locally with faster-whisper.", setup_required=[])

    def _transcribe_file(self, path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(path),
            language=self.settings.stt_language or "en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return " ".join(text.split())[:1000]

    def _load_model(self) -> Any:
        model_name = (self.settings.stt_model or "tiny.en").strip()
        if self._model is not None and self._model_name == model_name:
            return self._model

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=(self.settings.stt_compute_type or "int8").strip(),
        )
        self._model_name = model_name
        return self._model

    def _safe_suffix(self, filename: str) -> str:
        suffix = Path(filename or "voice.webm").suffix.lower()
        return suffix if suffix in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".oga"} else ".webm"
