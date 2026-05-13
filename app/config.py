from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    lm_studio_base_url: str = Field(default="http://127.0.0.1:1234/v1", alias="LM_STUDIO_BASE_URL")
    lm_studio_chat_model: str = Field(alias="LM_STUDIO_CHAT_MODEL")
    lm_studio_embed_model: str = Field(alias="LM_STUDIO_EMBED_MODEL")

    muserag_host: str = Field(default="0.0.0.0", alias="MUSERAG_HOST")
    muserag_port: int = Field(default=8000, alias="MUSERAG_PORT")
    muserag_chroma_dir: str = Field(default="./storage/chroma", alias="MUSERAG_CHROMA_DIR")
    muserag_collection: str = Field(default="museiq_knowledge", alias="MUSERAG_COLLECTION")
    muserag_pdf_path: str = Field(alias="MUSERAG_PDF_PATH")
    muserag_app_museum_json: str = Field(alias="MUSERAG_APP_MUSEUM_JSON")
    muserag_app_data_ts: str = Field(alias="MUSERAG_APP_DATA_TS")
    muserag_top_k: int = Field(default=4, alias="MUSERAG_TOP_K")
    muserag_log_interactions: bool = Field(default=True, alias="MUSERAG_LOG_INTERACTIONS")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.muserag_chroma_dir)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def pdf_path(self) -> Path:
        path = Path(self.muserag_pdf_path)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def museum_json_path(self) -> Path:
        path = Path(self.muserag_app_museum_json)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def app_data_ts_path(self) -> Path:
        path = Path(self.muserag_app_data_ts)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
