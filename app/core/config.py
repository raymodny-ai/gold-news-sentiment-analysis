"""
Application configuration settings.
"""
import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///F:/Financial Project/gold news/gold_news.db"  # Absolute path for SQLite
    redis_url: str = "redis://localhost:6379"  # Optional, can be disabled

    # API Keys
    newsapi_key: str = ""
    finnhub_api_key: str = ""
    yahoo_finance_api_key: str = ""

    # Application
    secret_key: str = "your-secret-key-change-in-production"
    debug: bool = True
    log_level: str = "INFO"

    # Celery (optional - can be disabled for simple setup)
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Models
    finbert_model_name: str = "ProsusAI/finbert"
    max_sequence_length: int = 512
    batch_size: int = 16

    # Sentiment Analysis
    vader_weight: float = 0.3
    textblob_weight: float = 0.2
    finbert_weight: float = 0.5

    # Time Parameters
    short_term_days: int = 7
    medium_term_days: int = 30
    long_term_days: int = 90
    decay_factor: float = 0.95

    # API
    api_host: str = "127.0.0.1"  # Changed from 0.0.0.0 to 127.0.0.1 for better compatibility
    api_port: int = 8000
    api_version: str = "v1"

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501

    # Rate Limiting
    rate_limit_per_minute: int = 100

    # Data Collection
    news_collection_interval_minutes: int = 60
    max_news_per_collection: int = 1000

    # News Categories
    news_categories: List[str] = [
        "macro_policy",
        "geopolitical",
        "economic_data",
        "market_sentiment",
        "central_bank"
    ]

    # Time Horizons
    time_horizons: List[str] = ["short", "medium", "long"]

    # Model Types
    model_types: List[str] = ["lstm", "xgboost", "ensemble"]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
