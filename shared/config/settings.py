"""Application settings and configuration management."""

from typing import Optional, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = Field(default="AmpereData", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API
    api_v1_str: str = Field(default="/api/v1", alias="API_V1_STR")
    backend_cors_origins: list[str] = Field(
        default=["http://localhost:3000"], alias="BACKEND_CORS_ORIGINS"
    )

    # Database
    database_url: str = Field(default="sqlite:///./test.db", alias="DATABASE_URL")
    database_test_url: Optional[str] = Field(default=None, alias="DATABASE_TEST_URL")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_url: str = Field(default="redis://localhost:6379/1", alias="REDIS_CACHE_URL")

    # Security
    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")
    jwt_secret_key: str = Field(default="dev-jwt-secret", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # File Storage
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    max_upload_size: str = Field(default="100MB", alias="MAX_UPLOAD_SIZE")

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/2", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(
        default="redis://localhost:6379/3", alias="CELERY_RESULT_BACKEND"
    )

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    prometheus_metrics_enabled: bool = Field(default=False, alias="PROMETHEUS_METRICS_ENABLED")

    @validator("backend_cors_origins", pre=True)
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, list[str]]) -> Union[list[str], str]:
        """Parse CORS origins from environment variable."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
