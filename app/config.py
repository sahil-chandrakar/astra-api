from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cerebras_api_key: str = ""
    tavily_api_key: str = ""
    semantic_scholar_api_key: str = ""
    frontend_origin: str = "http://localhost:3000"
    cerebras_model: str = "llama3.1-8b"
    voice_provider: str = "piper"
    piper_voice_id: str = "en_US-lessac-high"
    piper_cache_dir: str = ".cache/piper"
    piper_auto_download: bool = True
    piper_max_chars: int = 520
    piper_length_scale: float = 0.92
    piper_noise_scale: float = 0.667
    piper_noise_w_scale: float = 0.8
    piper_volume: float = 0.95
    reports_dir: str = "reports"
    data_dir: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env",
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
    def wants_piper(self) -> bool:
        return self.voice_provider.strip().lower() == "piper"


@lru_cache
def get_settings() -> Settings:
    return Settings()
