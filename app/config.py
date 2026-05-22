from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    cerebras_api_key: str = ""
    tavily_api_key: str = ""
    semantic_scholar_api_key: str = ""
    frontend_origin: str = "http://localhost:3000"
    cerebras_model: str = "zai-glm-4.7"
    cerebras_fast_model: str = "llama3.1-8b"
    cerebras_pro_model: str = ""
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "qwen/qwen3-coder-480b-a35b-instruct"
    voice_provider: str = "piper"
    piper_voice_id: str = "en_US-lessac-high"
    piper_cache_dir: str = ".cache/piper"
    piper_auto_download: bool = True
    piper_max_chars: int = 520
    piper_length_scale: float = 0.92
    piper_noise_scale: float = 0.667
    piper_noise_w_scale: float = 0.8
    piper_volume: float = 0.95
    stt_model: str = "tiny.en"
    stt_language: str = "en"
    stt_compute_type: str = "int8"
    stt_max_upload_bytes: int = 15_000_000
    reports_dir: str = "reports"
    data_dir: str = "data"
    astra_openrpa_exe: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_cerebras(self) -> bool:
        return bool(self.cerebras_api_key.strip())

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key.strip())

    @property
    def has_nvidia(self) -> bool:
        return bool(self.nvidia_api_key.strip())

    @property
    def wants_piper(self) -> bool:
        return self.voice_provider.strip().lower() == "piper"

    @property
    def resolved_cerebras_fast_model(self) -> str:
        return self.cerebras_fast_model.strip() or "llama3.1-8b"

    @property
    def resolved_cerebras_pro_model(self) -> str:
        return self.cerebras_pro_model.strip() or self.cerebras_model.strip() or "zai-glm-4.7"

    @property
    def cerebras_models(self) -> dict[str, str]:
        return {
            "fast": self.resolved_cerebras_fast_model,
            "pro": self.resolved_cerebras_pro_model,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
