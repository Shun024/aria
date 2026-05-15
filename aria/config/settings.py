"""
ARIA Configuration — all settings from environment variables.
No hardcoded values anywhere in the system.
"""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings


class ARIASettings(BaseSettings):
    # API Keys
    openai_api_key: SecretStr
    groq_api_key: SecretStr = SecretStr("")

    # Kafka
    kafka_bootstrap: str = "localhost:9092"
    kafka_news_topic: str = "aria.news"
    kafka_risk_topic: str = "aria.risk_events"
    kafka_consumer_group: str = "aria-consumer"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "aria_regulatory"

    # Database
    db_path: str = "data/aria.db"

    # Risk thresholds
    risk_watch_threshold: float = 40.0
    risk_alert_threshold: float = 60.0
    risk_escalate_threshold: float = 80.0

    # Model settings
    sentiment_model: str = "ProsusAI/finbert"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    dashboard_port: int = 8056

    # Monitoring
    drift_check_interval_seconds: int = 300
    mlflow_experiment: str = "aria"

    # Weights for risk score synthesis
    news_weight: float = 0.30
    market_weight: float = 0.40
    counterparty_weight: float = 0.30

    @field_validator("news_weight", "market_weight", "counterparty_weight")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("Weight must be between 0 and 1")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = ARIASettings()