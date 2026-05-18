import asyncio
import hashlib
import importlib.util
import os
import re
import threading
import wave
from pathlib import Path

from app.config import Settings
from app.models import VoiceStatusResponse


class VoiceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = Path(settings.piper_cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = Path.cwd() / self.cache_dir
        self.voice_dir = self.cache_dir / "voices"
        self.audio_dir = self.cache_dir / "audio"
        self._voice = None
        self._voice_id = ""
        self._thread_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def status(self) -> VoiceStatusResponse:
        setup_required: list[str] = []
        package_available = importlib.util.find_spec("piper") is not None
        if not self.settings.wants_piper:
            setup_required.append("VOICE_PROVIDER=piper")
        if not package_available:
            setup_required.append("piper-tts")

        model_path = self._model_path(self.settings.piper_voice_id)
        cached = model_path.exists() and self._config_path(self.settings.piper_voice_id).exists()
        if not cached and not self.settings.piper_auto_download:
            setup_required.append(f"{self.settings.piper_voice_id}.onnx")

        enabled = self.settings.wants_piper and package_available and (cached or self.settings.piper_auto_download)
        message = "Piper local voice is ready." if enabled else "Piper local voice needs setup."
        return VoiceStatusResponse(
            enabled=enabled,
            provider="piper",
            voice=self.settings.piper_voice_id,
            cached=cached,
            loaded=self._voice is not None and self._voice_id == self.settings.piper_voice_id,
            setup_required=setup_required,
            message=message,
        )

    async def warm_up(self, voice_id: str | None = None) -> VoiceStatusResponse:
        voice_id = (voice_id or self.settings.piper_voice_id).strip()
        if not self.status().enabled:
            return self.status()
        await self.synthesize("Ready.", voice_id)
        return self.status()

    async def synthesize(self, text: str, voice_id: str | None = None) -> tuple[bytes, str, bool]:
        voice_id = (voice_id or self.settings.piper_voice_id).strip()
        clean_text = self._normalize_text(text)
        if not clean_text:
            raise ValueError("No text to synthesize.")

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.audio_dir / f"{self._cache_key(clean_text, voice_id)}.wav"
        if cache_path.exists() and cache_path.stat().st_size > 44:
            return cache_path.read_bytes(), voice_id, True

        async with self._async_lock:
            if cache_path.exists() and cache_path.stat().st_size > 44:
                return cache_path.read_bytes(), voice_id, True
            await asyncio.to_thread(self._synthesize_to_file, clean_text, voice_id, cache_path)
            return cache_path.read_bytes(), voice_id, False

    def _synthesize_to_file(self, text: str, voice_id: str, cache_path: Path) -> None:
        with self._thread_lock:
            self._ensure_voice_files(voice_id)
            voice = self._load_voice(voice_id)
            temp_path = cache_path.with_suffix(".tmp.wav")
            if temp_path.exists():
                temp_path.unlink()

            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=self.settings.piper_length_scale,
                noise_scale=self.settings.piper_noise_scale,
                noise_w_scale=self.settings.piper_noise_w_scale,
                volume=self.settings.piper_volume,
            )

            try:
                with wave.open(str(temp_path), "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                os.replace(temp_path, cache_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    def _load_voice(self, voice_id: str):
        if self._voice is not None and self._voice_id == voice_id:
            return self._voice

        from piper import PiperVoice

        self._voice = PiperVoice.load(
            self._model_path(voice_id),
            config_path=self._config_path(voice_id),
            download_dir=self.voice_dir,
        )
        self._voice_id = voice_id
        return self._voice

    def _ensure_voice_files(self, voice_id: str) -> None:
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_path(voice_id)
        config_path = self._config_path(voice_id)
        if model_path.exists() and model_path.stat().st_size > 0 and config_path.exists() and config_path.stat().st_size > 0:
            return
        if not self.settings.piper_auto_download:
            raise RuntimeError(f"Piper voice files are missing for {voice_id}.")

        from piper.download_voices import download_voice

        download_voice(voice_id, self.voice_dir)

    def _model_path(self, voice_id: str) -> Path:
        return self.voice_dir / f"{voice_id}.onnx"

    def _config_path(self, voice_id: str) -> Path:
        return self.voice_dir / f"{voice_id}.onnx.json"

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.replace("*", "")).strip()
        return text[: self.settings.piper_max_chars]

    def _cache_key(self, text: str, voice_id: str) -> str:
        parts = [
            voice_id,
            str(self.settings.piper_length_scale),
            str(self.settings.piper_noise_scale),
            str(self.settings.piper_noise_w_scale),
            str(self.settings.piper_volume),
            text,
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
