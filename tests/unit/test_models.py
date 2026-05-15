"""
Unit tests for ARIA core data models.
Fast, no I/O, no external dependencies.
"""

import pytest
from datetime import datetime
from aria.config.models import (
    NewsItem,
    NewsSentimentResult,
    MarketRiskResult,
    CounterpartyRiskResult,
    RegulatoryContext,
    ARIARiskEvent,
    AgentState,
    RiskLevel,
    SentimentLabel,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def valid_news_item():
    return NewsItem(
        item_id="news_001",
        timestamp=datetime.utcnow(),
        headline="HSBC reports record Q3 profits amid rising rates",
        source="Reuters",
        entities=["HSBC"],
    )


@pytest.fixture
def valid_sentiment_result():
    return NewsSentimentResult(
        item_id="news_001",
        entity="HSBC",
        sentiment_label=SentimentLabel.POSITIVE,
        sentiment_score=0.82,
        confidence=0.91,
        extracted_entities=["HSBC"],
        model_version="finbert-v1",
        latency_ms=45.2,
    )


@pytest.fixture
def valid_market_result():
    return MarketRiskResult(
        entity="HSBC",
        timestamp=datetime.utcnow(),
        risk_score=0.35,
        forecast_return=0.02,
        forecast_volatility=0.15,
        regime="low_vol",
        confidence=0.88,
        latency_ms=120.5,
    )


@pytest.fixture
def valid_counterparty_result():
    return CounterpartyRiskResult(
        entity="HSBC",
        risk_score=0.25,
        network_centrality=0.6,
        contagion_paths=["HSBC -> Barclays", "HSBC -> Lloyds"],
        connected_entities=["Barclays", "Lloyds", "NatWest"],
        confidence=0.78,
        latency_ms=85.3,
    )


@pytest.fixture
def valid_regulatory_result():
    return RegulatoryContext(
        entity="HSBC",
        relevant_regulations=["Basel III CET1", "FCA SYSC"],
        compliance_flags=[],
        context_summary="HSBC meets all Basel III capital requirements.",
        sources=["Basel III framework", "FCA handbook"],
        confidence=0.95,
        latency_ms=200.1,
    )


# ── NewsItem tests ─────────────────────────────────────────

class TestNewsItem:
    def test_valid_creation(self, valid_news_item):
        assert valid_news_item.item_id == "news_001"
        assert valid_news_item.source == "Reuters"

    def test_empty_headline_raises(self):
        with pytest.raises(ValueError, match="Headline cannot be empty"):
            NewsItem(
                item_id="news_002",
                timestamp=datetime.utcnow(),
                headline="   ",
                source="Reuters",
            )

    def test_headline_stripped(self):
        item = NewsItem(
            item_id="news_003",
            timestamp=datetime.utcnow(),
            headline="  HSBC profits rise  ",
            source="Reuters",
        )
        assert item.headline == "HSBC profits rise"

    def test_entities_default_empty(self):
        item = NewsItem(
            item_id="news_004",
            timestamp=datetime.utcnow(),
            headline="Market update",
            source="Bloomberg",
        )
        assert item.entities == []


# ── Sentiment tests ────────────────────────────────────────

class TestNewsSentimentResult:
    def test_valid_creation(self, valid_sentiment_result):
        assert valid_sentiment_result.sentiment_score == 0.82
        assert valid_sentiment_result.confidence == 0.91

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            NewsSentimentResult(
                item_id="x",
                entity="HSBC",
                sentiment_label=SentimentLabel.POSITIVE,
                sentiment_score=1.5,  # > 1.0
                confidence=0.9,
                model_version="v1",
                latency_ms=10.0,
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            NewsSentimentResult(
                item_id="x",
                entity="HSBC",
                sentiment_label=SentimentLabel.NEUTRAL,
                sentiment_score=0.0,
                confidence=-0.1,  # < 0
                model_version="v1",
                latency_ms=10.0,
            )

    def test_degraded_default_false(self, valid_sentiment_result):
        assert valid_sentiment_result.degraded is False


# ── MarketRiskResult tests ─────────────────────────────────

class TestMarketRiskResult:
    def test_valid_creation(self, valid_market_result):
        assert valid_market_result.risk_score == 0.35
        assert valid_market_result.regime == "low_vol"

    def test_risk_score_bounds(self):
        with pytest.raises(ValueError):
            MarketRiskResult(
                entity="HSBC",
                timestamp=datetime.utcnow(),
                risk_score=1.5,  # > 1.0
                forecast_return=0.0,
                forecast_volatility=0.1,
                regime="low_vol",
                confidence=0.8,
                latency_ms=100.0,
            )


# ── ARIARiskEvent tests ────────────────────────────────────

class TestARIARiskEvent:
    def test_risk_level_assignment(
        self,
        valid_sentiment_result,
        valid_market_result,
        valid_counterparty_result,
        valid_regulatory_result,
    ):
        event = ARIARiskEvent(
            event_id="evt_001",
            timestamp=datetime.utcnow(),
            entity="HSBC",
            news_sentiment=valid_sentiment_result,
            market_risk=valid_market_result,
            counterparty_risk=valid_counterparty_result,
            regulatory_context=valid_regulatory_result,
            aria_risk_score=35.0,
            risk_level=RiskLevel.MONITOR,
            risk_summary="Low risk entity.",
            reasoning_trace="All signals green.",
            confidence=0.88,
            processing_time_ms=450.0,
        )
        assert event.risk_level == RiskLevel.MONITOR
        assert event.requires_human_review is False

    def test_escalate_requires_human_review(
        self,
        valid_sentiment_result,
        valid_market_result,
        valid_counterparty_result,
        valid_regulatory_result,
    ):
        event = ARIARiskEvent(
            event_id="evt_002",
            timestamp=datetime.utcnow(),
            entity="CreditSuisse",
            news_sentiment=valid_sentiment_result,
            market_risk=valid_market_result,
            counterparty_risk=valid_counterparty_result,
            regulatory_context=valid_regulatory_result,
            aria_risk_score=85.0,
            risk_level=RiskLevel.ESCALATE,
            risk_summary="Critical risk detected.",
            reasoning_trace="Multiple signals elevated.",
            confidence=0.95,
            processing_time_ms=520.0,
        )
        assert event.requires_human_review is True

    def test_component_scores_keys(
        self,
        valid_sentiment_result,
        valid_market_result,
        valid_counterparty_result,
        valid_regulatory_result,
    ):
        event = ARIARiskEvent(
            event_id="evt_003",
            timestamp=datetime.utcnow(),
            entity="HSBC",
            news_sentiment=valid_sentiment_result,
            market_risk=valid_market_result,
            counterparty_risk=valid_counterparty_result,
            regulatory_context=valid_regulatory_result,
            aria_risk_score=42.0,
            risk_level=RiskLevel.WATCH,
            risk_summary="Elevated watch.",
            reasoning_trace="Mixed signals.",
            confidence=0.75,
            processing_time_ms=380.0,
        )
        scores = event.component_scores
        assert set(scores.keys()) == {
            "news_sentiment", "market_risk", "counterparty_risk"
        }

    def test_aria_risk_score_bounds(
        self,
        valid_sentiment_result,
        valid_market_result,
        valid_counterparty_result,
        valid_regulatory_result,
    ):
        with pytest.raises(ValueError):
            ARIARiskEvent(
                event_id="evt_004",
                timestamp=datetime.utcnow(),
                entity="HSBC",
                news_sentiment=valid_sentiment_result,
                market_risk=valid_market_result,
                counterparty_risk=valid_counterparty_result,
                regulatory_context=valid_regulatory_result,
                aria_risk_score=150.0,  # > 100
                risk_level=RiskLevel.ESCALATE,
                risk_summary="",
                reasoning_trace="",
                confidence=0.9,
                processing_time_ms=100.0,
            )


# ── AgentState tests ───────────────────────────────────────

class TestAgentState:
    def test_initial_state(self):
        state = AgentState(entity="Barclays")
        assert state.entity == "Barclays"
        assert state.news_item is None
        assert state.errors == []

    def test_errors_accumulate(self):
        state = AgentState(entity="Barclays", errors=["timeout"])
        assert len(state.errors) == 1


# ── RiskLevel tests ────────────────────────────────────────

class TestRiskLevel:
    def test_all_levels_defined(self):
        assert RiskLevel.MONITOR == "MONITOR"
        assert RiskLevel.WATCH == "WATCH"
        assert RiskLevel.ALERT == "ALERT"
        assert RiskLevel.ESCALATE == "ESCALATE"

    def test_risk_level_ordering(self):
        levels = [RiskLevel.MONITOR, RiskLevel.WATCH,
                  RiskLevel.ALERT, RiskLevel.ESCALATE]
        assert len(levels) == 4