from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERSEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    oracle_user: str = Field(default="papersearch", description="Oracle schema / user name")
    oracle_password: str = Field(default="PaperSearch_App_1")
    oracle_dsn: str = Field(default="localhost:1521/FREEPDB1", description="Easy Connect or TNS")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="384-d MiniLM model; dimension must match VECTOR(384, FLOAT32)",
    )
    embedding_dimension: int = Field(default=384, ge=8, le=4096)

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = Field(
        default="http://localhost:8000,http://127.0.0.1:8000",
        description="Comma-separated origins for browser demo",
    )

    use_approximate_fetch: bool = Field(
        default=True,
        description="Use FETCH APPROXIMATE with HNSW (requires vector index)",
    )

    import_max_papers: int = Field(
        default=1200,
        ge=1,
        le=2_000_000,
        description="Max papers: import-kaggle default; seed truncates JSON to first N rows",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, v: object) -> str:
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return str(v)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
