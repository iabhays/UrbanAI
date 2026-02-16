"""
SENTIENTCITY AI - Core Settings Module
Centralized configuration management using pydantic-settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    """Kafka connection settings."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    bootstrap_servers: str = Field(default="localhost:9092")
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: str | None = Field(default=None)
    sasl_username: str | None = Field(default=None)
    sasl_password: SecretStr | None = Field(default=None)
    consumer_group: str = Field(default="sentientcity")
    auto_offset_reset: str = Field(default="latest")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: SecretStr | None = Field(default=None)
    ssl: bool = Field(default=False)
    max_connections: int = Field(default=100)

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class DatabaseSettings(BaseSettings):
    """PostgreSQL database settings."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="sentientcity")
    user: str = Field(default="postgres")
    password: SecretStr = Field(default=SecretStr("postgres"))
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)

    @property
    def async_url(self) -> str:
        """Generate async database URL."""
        return (
            f"postgresql+asyncpg://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_url(self) -> str:
        """Generate sync database URL."""
        return (
            f"postgresql://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.name}"
        )


class InferenceSettings(BaseSettings):
    """ML inference settings."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    device: str = Field(default="cuda:0")
    batch_size: int = Field(default=1)
    fp16: bool = Field(default=True)
    tensorrt: bool = Field(default=False)
    model_path: str = Field(default="/models")
    confidence_threshold: float = Field(default=0.5)
    nms_threshold: float = Field(default=0.45)


class LLMSettings(BaseSettings):
    """LLM integration settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Literal["openai", "anthropic", "local"] = Field(default="openai")
    api_key: SecretStr | None = Field(default=None)
    model: str = Field(default="gpt-4")
    max_tokens: int = Field(default=500)
    temperature: float = Field(default=0.3)
    base_url: str | None = Field(default=None)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    service_name: str = Field(default="sentientcity")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: list[str] = Field(default=["*"])

    # Security
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_minutes: int = Field(default=60)

    # Sub-settings
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
