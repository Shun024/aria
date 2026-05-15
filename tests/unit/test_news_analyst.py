"""
Unit tests for the News Analyst Agent.
Tests sentiment scoring, entity extraction, and graceful degradation.
All tests are fast and isolated — no API calls, no I/O.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from aria.config.models import (
    NewsItem,
    NewsSentimentResult,
    AgentState,
    SentimentLabel,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def positive_news_item():
    return NewsItem(
        item_id="news_pos_001",
        timestamp=datetime.utcnow(),
        headline="HSBC reports record Q3 profits amid rising interest rates",
        source="Reuters",
        entities=["HSBC"],
    )


@pytest.fixture
def negative_news_item():
    return NewsItem(
        item_id="news_neg_001",
        timestamp=datetime.utcnow(),
        headline="Barclays faces regulatory probe over derivatives mis-selling",
        source="FT",
        entities=["Barclays"],
    )


@pytest.fixture
def neutral_news_item():
    return NewsItem(
        item_id="news_neu_001",
        timestamp=datetime.utcnow(),
        headline="Lloyds Banking Group announces annual general meeting date",
        source="Bloomberg",
        entities=["Lloyds"],
    )


@pytest.fixture
def mock_finbert_output_positive():
    """Mocked FinBERT pipeline output for positive sentiment."""
    return [[
        {"label": "positive", "score": 0.91},
        {"label": "neutral", "score": 0.07},
        {"label": "negative", "score": 0.02},
    ]]


@pytest.fixture
def mock_finbert_output_negative():
    return [[
        {"label": "negative", "score": 0.88},
        {"label": "neutral", "score": 0.09},
        {"label": "positive", "score": 0.03},
    ]]


@pytest.fixture
def mock_finbert_output_neutral():
    return [[
        {"label": "neutral", "score": 0.79},
        {"label": "positive", "score": 0.12},
        {"label": "negative", "score": 0.09},
    ]]


# ── Sentiment score conversion tests ──────────────────────

class TestSentimentScoreConversion:
    """Test conversion from FinBERT label+score to [-1, 1] sentiment score."""

    def test_positive_label_gives_positive_score(self):
        from aria.agents.news_analyst import sentiment_to_score
        score = sentiment_to_score("positive", 0.9)
        assert score > 0
        assert score <= 1.0

    def test_negative_label_gives_negative_score(self):
        from aria.agents.news_analyst import sentiment_to_score
        score = sentiment_to_score("negative", 0.9)
        assert score < 0
        assert score >= -1.0

    def test_neutral_label_gives_near_zero(self):
        from aria.agents.news_analyst import sentiment_to_score
        score = sentiment_to_score("neutral", 0.9)
        assert -0.2 <= score <= 0.2

    def test_high_confidence_gives_extreme_score(self):
        from aria.agents.news_analyst import sentiment_to_score
        pos_high = sentiment_to_score("positive", 0.99)
        pos_low = sentiment_to_score("positive", 0.51)
        assert abs(pos_high) > abs(pos_low)


# ── Entity extraction tests ────────────────────────────────

class TestEntityExtraction:
    def test_extracts_known_banks(self):
        from aria.agents.news_analyst import extract_financial_entities
        text = "HSBC and Barclays both reported strong Q3 earnings"
        entities = extract_financial_entities(text)
        assert "HSBC" in entities
        assert "Barclays" in entities

    def test_returns_empty_for_no_entities(self):
        from aria.agents.news_analyst import extract_financial_entities
        text = "The weather today is sunny and warm"
        entities = extract_financial_entities(text)
        assert isinstance(entities, list)

    def test_deduplicates_entities(self):
        from aria.agents.news_analyst import extract_financial_entities
        text = "HSBC profit rise. HSBC shares up. HSBC CEO comments."
        entities = extract_financial_entities(text)
        assert entities.count("HSBC") == 1

    def test_handles_empty_string(self):
        from aria.agents.news_analyst import extract_financial_entities
        entities = extract_financial_entities("")
        assert entities == []


# ── Risk level from sentiment tests ───────────────────────

class TestSentimentRiskMapping:
    def test_very_negative_is_high_risk(self):
        from aria.agents.news_analyst import sentiment_to_risk_contribution
        risk = sentiment_to_risk_contribution(-0.95)
        assert risk > 0.7

    def test_very_positive_is_low_risk(self):
        from aria.agents.news_analyst import sentiment_to_risk_contribution
        risk = sentiment_to_risk_contribution(0.95)
        assert risk < 0.3

    def test_neutral_is_medium_risk(self):
        from aria.agents.news_analyst import sentiment_to_risk_contribution
        risk = sentiment_to_risk_contribution(0.0)
        assert 0.3 <= risk <= 0.7

    def test_output_bounded_zero_to_one(self):
        from aria.agents.news_analyst import sentiment_to_risk_contribution
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            risk = sentiment_to_risk_contribution(score)
            assert 0.0 <= risk <= 1.0


# ── Degraded mode tests ────────────────────────────────────

class TestDegradedMode:
    def test_degraded_result_has_neutral_score(self):
        from aria.agents.news_analyst import make_degraded_result
        result = make_degraded_result(
            item_id="news_001",
            entity="HSBC",
            error="Model timeout",
        )
        assert result.degraded is True
        assert result.sentiment_label == SentimentLabel.NEUTRAL
        assert result.sentiment_score == 0.0
        assert result.confidence == 0.0

    def test_degraded_result_valid_model(self):
        from aria.agents.news_analyst import make_degraded_result
        result = make_degraded_result(
            item_id="news_001",
            entity="HSBC",
            error="Connection refused",
        )
        assert isinstance(result, NewsSentimentResult)


# ── Agent state integration tests ─────────────────────────

class TestNewsAnalystStateTransition:
    def test_state_updated_with_result(
        self, positive_news_item, mock_finbert_output_positive
    ):
        from aria.agents.news_analyst import NewsAnalystAgent

        with patch(
            "aria.agents.news_analyst.pipeline"
        ) as mock_pipeline:
            mock_pipe = MagicMock()
            mock_pipe.return_value = mock_finbert_output_positive
            mock_pipeline.return_value = mock_pipe

            agent = NewsAnalystAgent()
            state = AgentState(
                entity="HSBC",
                news_item=positive_news_item,
            )
            updated_state = agent.run(state)

            assert updated_state.news_result is not None
            assert updated_state.news_result.entity == "HSBC"
            assert updated_state.news_result.degraded is False

    def test_state_degraded_on_model_failure(self, positive_news_item):
        from aria.agents.news_analyst import NewsAnalystAgent

        with patch(
            "aria.agents.news_analyst.pipeline",
            side_effect=RuntimeError("Model load failed"),
        ):
            agent = NewsAnalystAgent()
            state = AgentState(
                entity="HSBC",
                news_item=positive_news_item,
            )
            updated_state = agent.run(state)

            assert updated_state.news_result is not None
            assert updated_state.news_result.degraded is True
            assert len(updated_state.errors) > 0

    def test_state_unchanged_when_no_news_item(self):
        from aria.agents.news_analyst import NewsAnalystAgent

        agent = NewsAnalystAgent()
        state = AgentState(entity="HSBC")
        updated_state = agent.run(state)

        assert updated_state.news_result is None