"""
ARIA Core Data Models
All data flowing through the system is typed via Pydantic.
No raw dicts between components.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    MONITOR = "MONITOR"       # Score 0-40: watch passively
    WATCH = "WATCH"           # Score 40-60: increased attention
    ALERT = "ALERT"           # Score 60-80: analyst notification
    ESCALATE = "ESCALATE"     # Score 80-100: immediate human review


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class NewsItem(BaseModel):
    """Raw news item from the stream."""
    item_id: str
    timestamp: datetime
    headline: str
    body: Optional[str] = None
    source: str
    entities: list[str] = Field(default_factory=list)
    url: Optional[str] = None

    @field_validator("headline")
    @classmethod
    def headline_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Headline cannot be empty")
        return v.strip()


class NewsSentimentResult(BaseModel):
    """Output of the News Analyst Agent."""
    item_id: str
    entity: str
    sentiment_label: SentimentLabel
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_entities: list[str] = Field(default_factory=list)
    model_version: str
    latency_ms: float
    degraded: bool = False


class MarketRiskResult(BaseModel):
    """Output of the Market Risk Agent."""
    entity: str
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=1.0)
    forecast_return: float
    forecast_volatility: float
    regime: str  # "low_vol", "high_vol", "crisis"
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    degraded: bool = False


class CounterpartyRiskResult(BaseModel):
    """Output of the Counterparty Risk Agent."""
    entity: str
    risk_score: float = Field(ge=0.0, le=1.0)
    network_centrality: float = Field(ge=0.0, le=1.0)
    contagion_paths: list[str] = Field(default_factory=list)
    connected_entities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    degraded: bool = False


class RegulatoryContext(BaseModel):
    """Output of the Regulatory Context Agent."""
    entity: str
    relevant_regulations: list[str] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)
    context_summary: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    degraded: bool = False


class ARIARiskEvent(BaseModel):
    """
    Final unified risk assessment.
    The primary output of the ARIA system.
    """
    event_id: str
    timestamp: datetime
    entity: str

    # Agent outputs
    news_sentiment: NewsSentimentResult
    market_risk: MarketRiskResult
    counterparty_risk: CounterpartyRiskResult
    regulatory_context: RegulatoryContext

    # Synthesised risk
    aria_risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    risk_summary: str
    reasoning_trace: str
    confidence: float = Field(ge=0.0, le=1.0)

    # System metadata
    processing_time_ms: float
    any_degraded: bool = False

    @property
    def requires_human_review(self) -> bool:
        return self.risk_level == RiskLevel.ESCALATE

    @property
    def component_scores(self) -> dict[str, float]:
        return {
            "news_sentiment": abs(self.news_sentiment.sentiment_score),
            "market_risk": self.market_risk.risk_score,
            "counterparty_risk": self.counterparty_risk.risk_score,
        }


class AgentState(BaseModel):
    """
    LangGraph state object — passed between all agents.
    Immutable between steps (each agent returns updated copy).
    """
    entity: str
    news_item: Optional[NewsItem] = None
    news_result: Optional[NewsSentimentResult] = None
    market_result: Optional[MarketRiskResult] = None
    counterparty_result: Optional[CounterpartyRiskResult] = None
    regulatory_result: Optional[RegulatoryContext] = None
    final_risk: Optional[ARIARiskEvent] = None
    errors: list[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None