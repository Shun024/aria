"""
ARIA News Analyst Agent
Scores financial news sentiment using FinBERT and extracts entities.

Design decisions:
- FinBERT over general BERT: domain-adapted for financial text
- Graceful degradation: returns neutral score on failure, never crashes
- Pure function helpers: sentiment_to_score, extract_financial_entities
  are testable without instantiating the full agent
"""

from __future__ import annotations

import re
import time
import structlog
from datetime import datetime

from aria.config.models import (
    AgentState,
    NewsSentimentResult,
    SentimentLabel,
)

logger = structlog.get_logger(__name__)

# Module-level import makes it patchable in tests
try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore

FINANCIAL_ENTITIES = [
    "HSBC", "Barclays", "Lloyds", "NatWest", "Standard Chartered",
    "Santander", "Halifax", "Nationwide", "TSB", "Metro Bank",
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "Deutsche Bank",
    "Credit Suisse", "UBS", "BNP Paribas", "Societe Generale",
    "BlackRock", "Vanguard", "Fidelity", "Schroders",
    "FCA", "PRA", "Bank of England", "ECB", "Fed", "Federal Reserve",
]

MODEL_VERSION = "finbert-v1.0"


def sentiment_to_score(label: str, confidence: float) -> float:
    """Convert FinBERT label + confidence to [-1, 1] sentiment score."""
    if label == "positive":
        return round(confidence, 4)
    elif label == "negative":
        return round(-confidence, 4)
    else:
        return round((confidence - 1.0) * 0.2, 4)


def extract_financial_entities(text: str) -> list[str]:
    """Extract known financial entities from text."""
    if not text:
        return []
    found = []
    for entity in FINANCIAL_ENTITIES:
        pattern = r"\b" + re.escape(entity) + r"\b"
        if re.search(pattern, text):
            found.append(entity)
    seen = set()
    deduped = []
    for e in found:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped


def sentiment_to_risk_contribution(sentiment_score: float) -> float:
    """Convert sentiment score [-1, 1] to risk contribution [0, 1]."""
    risk = (1.0 - sentiment_score) / 2.0
    return round(max(0.0, min(1.0, risk)), 4)


def make_degraded_result(
    item_id: str,
    entity: str,
    error: str,
) -> NewsSentimentResult:
    """Return neutral degraded result when model fails."""
    logger.warning(
        "news_analyst.degraded",
        item_id=item_id,
        entity=entity,
        error=error,
    )
    return NewsSentimentResult(
        item_id=item_id,
        entity=entity,
        sentiment_label=SentimentLabel.NEUTRAL,
        sentiment_score=0.0,
        confidence=0.0,
        extracted_entities=[],
        model_version=MODEL_VERSION,
        latency_ms=0.0,
        degraded=True,
    )


class NewsAnalystAgent:
    """
    News Analyst Agent — scores financial news sentiment.
    Wraps FinBERT with lazy loading and graceful degradation.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load FinBERT pipeline — only on first call."""
        if self._pipeline is None:
            if pipeline is None:
                raise ImportError("transformers not installed")
            logger.info("news_analyst.loading_model", model=self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("news_analyst.model_loaded", model=self.model_name)

    def _score_text(self, text: str) -> tuple[str, float]:
        """Run FinBERT on text and return (label, confidence)."""
        results = self._pipeline(text)
        if isinstance(results[0], list):
            results = results[0]
        best = max(results, key=lambda x: x["score"])
        return best["label"].lower(), best["score"]

    def run(self, state: AgentState) -> AgentState:
        """
        Process news item in state and return updated state.
        AgentState (with news_item) -> AgentState (with news_result)
        """
        if state.news_item is None:
            logger.debug("news_analyst.no_news_item", entity=state.entity)
            return state

        t0 = time.time()
        item = state.news_item

        try:
            self._load_pipeline()
            label, confidence = self._score_text(item.headline)
            sentiment_score = sentiment_to_score(label, confidence)
            sentiment_label = (
                SentimentLabel(label)
                if label in ["positive", "negative", "neutral"]
                else SentimentLabel.NEUTRAL
            )
            extracted = extract_financial_entities(
                f"{item.headline} {item.body or ''}"
            )
            latency_ms = (time.time() - t0) * 1000

            result = NewsSentimentResult(
                item_id=item.item_id,
                entity=state.entity,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                confidence=round(confidence, 4),
                extracted_entities=extracted,
                model_version=MODEL_VERSION,
                latency_ms=round(latency_ms, 2),
                degraded=False,
            )

            logger.info(
                "news_analyst.scored",
                entity=state.entity,
                sentiment=label,
                score=sentiment_score,
                latency_ms=round(latency_ms, 2),
            )

            return AgentState(**{**state.model_dump(), "news_result": result})

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(
                "news_analyst.error",
                entity=state.entity,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            result = make_degraded_result(
                item_id=item.item_id,
                entity=state.entity,
                error=str(e),
            )
            return AgentState(**{
                **state.model_dump(),
                "news_result": result,
                "errors": state.errors + [f"news_analyst: {str(e)}"],
            })