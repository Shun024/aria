"""
Unit tests for ARIA FastAPI endpoints.
Uses TestClient — no real server, no external dependencies.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from aria.config.models import (
    ARIARiskEvent,
    RiskLevel,
    SentimentLabel,
    NewsSentimentResult,
    MarketRiskResult,
    CounterpartyRiskResult,
    RegulatoryContext,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_risk_event():
    return ARIARiskEvent(
        event_id="evt_test_001",
        timestamp=datetime.utcnow(),
        entity="HSBC",
        news_sentiment=NewsSentimentResult(
            item_id="n1", entity="HSBC",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.75, confidence=0.88,
            model_version="finbert-v1", latency_ms=45.0,
        ),
        market_risk=MarketRiskResult(
            entity="HSBC", timestamp=datetime.utcnow(),
            risk_score=0.25, forecast_return=0.06,
            forecast_volatility=0.12, regime="low_vol",
            confidence=0.85, latency_ms=30.0,
        ),
        counterparty_risk=CounterpartyRiskResult(
            entity="HSBC", risk_score=0.20,
            network_centrality=0.4, confidence=0.80,
            latency_ms=25.0,
        ),
        regulatory_context=RegulatoryContext(
            entity="HSBC", compliance_flags=[],
            context_summary="No issues.",
            confidence=0.95, latency_ms=50.0,
        ),
        aria_risk_score=22.5,
        risk_level=RiskLevel.MONITOR,
        risk_summary="Low risk entity.",
        reasoning_trace="All signals green.",
        confidence=0.88,
        processing_time_ms=180.0,
    )


@pytest.fixture
def client(mock_risk_event):
    """FastAPI test client with mocked pipeline."""
    with patch("aria.serving.api.run_aria_pipeline",
               return_value=mock_risk_event):
        from aria.serving.api import app
        return TestClient(app)


# ── Health endpoint tests ──────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    def test_health_returns_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert "version" in data


# ── Risk assessment endpoint tests ────────────────────────

class TestRiskAssessmentEndpoint:
    def test_assess_returns_200(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        assert response.status_code == 200

    def test_assess_returns_risk_score(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        data = response.json()
        assert "aria_risk_score" in data
        assert 0 <= data["aria_risk_score"] <= 100

    def test_assess_returns_risk_level(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        data = response.json()
        assert data["risk_level"] in ["MONITOR", "WATCH", "ALERT", "ESCALATE"]

    def test_assess_returns_entity(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        data = response.json()
        assert data["entity"] == "HSBC"

    def test_assess_missing_entity_returns_422(self, client):
        response = client.post("/assess", json={})
        assert response.status_code == 422

    def test_assess_with_headline(self, client):
        response = client.post(
            "/assess",
            json={
                "entity": "HSBC",
                "headline": "HSBC reports record profits",
                "source": "Reuters",
            },
        )
        assert response.status_code == 200

    def test_assess_returns_reasoning_trace(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        data = response.json()
        assert "reasoning_trace" in data
        assert len(data["reasoning_trace"]) > 0

    def test_assess_returns_processing_time(self, client):
        response = client.post(
            "/assess",
            json={"entity": "HSBC"},
        )
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0


# ── Batch assessment tests ─────────────────────────────────

class TestBatchAssessmentEndpoint:
    def test_batch_returns_200(self, client):
        response = client.post(
            "/assess/batch",
            json={"entities": ["HSBC", "Barclays"]},
        )
        assert response.status_code == 200

    def test_batch_returns_results_for_all_entities(self, client):
        response = client.post(
            "/assess/batch",
            json={"entities": ["HSBC", "Barclays", "Lloyds"]},
        )
        data = response.json()
        assert len(data["results"]) == 3

    def test_batch_empty_entities_returns_422(self, client):
        response = client.post(
            "/assess/batch",
            json={"entities": []},
        )
        assert response.status_code == 422

    def test_batch_returns_total_count(self, client):
        response = client.post(
            "/assess/batch",
            json={"entities": ["HSBC", "Barclays"]},
        )
        data = response.json()
        assert data["total"] == 2


# ── Entity list endpoint ───────────────────────────────────

class TestEntitiesEndpoint:
    def test_entities_returns_200(self, client):
        response = client.get("/entities")
        assert response.status_code == 200

    def test_entities_returns_list(self, client):
        response = client.get("/entities")
        data = response.json()
        assert isinstance(data["entities"], list)
        assert len(data["entities"]) > 0

    def test_entities_includes_major_banks(self, client):
        response = client.get("/entities")
        data = response.json()
        entities = data["entities"]
        assert "HSBC" in entities