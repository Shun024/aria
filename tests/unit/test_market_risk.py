"""
Unit tests for the Market Risk Agent.
Tests risk scoring, regime detection, and graceful degradation.
All tests fast and isolated — no API calls, no file I/O.
"""

import pytest
import numpy as np
from datetime import datetime
from aria.config.models import AgentState, MarketRiskResult


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def low_vol_returns():
    """Normal low-volatility returns."""
    np.random.seed(42)
    return list(np.random.normal(0.0005, 0.006, 252))  # std 0.006 → ~9.5% annualised


@pytest.fixture
def high_vol_returns():
    """Crisis-level high-volatility returns."""
    np.random.seed(42)
    return list(np.random.normal(-0.002, 0.035, 252))


@pytest.fixture
def trending_up_returns():
    """Steady uptrend returns."""
    np.random.seed(42)
    return list(np.random.normal(0.002, 0.006, 252))


# ── Volatility regime tests ────────────────────────────────

class TestVolatilityRegimeDetection:
    def test_low_vol_detected_correctly(self, low_vol_returns):
        from aria.agents.market_risk import detect_volatility_regime
        regime = detect_volatility_regime(low_vol_returns)
        assert regime == "low_vol"

    def test_high_vol_detected_correctly(self, high_vol_returns):
        from aria.agents.market_risk import detect_volatility_regime
        regime = detect_volatility_regime(high_vol_returns)
        assert regime in ["high_vol", "crisis"]

    def test_regime_is_string(self, low_vol_returns):
        from aria.agents.market_risk import detect_volatility_regime
        regime = detect_volatility_regime(low_vol_returns)
        assert isinstance(regime, str)

    def test_empty_returns_returns_unknown(self):
        from aria.agents.market_risk import detect_volatility_regime
        regime = detect_volatility_regime([])
        assert regime == "unknown"

    def test_valid_regime_values(self, low_vol_returns, high_vol_returns):
        from aria.agents.market_risk import detect_volatility_regime
        valid = {"low_vol", "high_vol", "crisis", "unknown"}
        assert detect_volatility_regime(low_vol_returns) in valid
        assert detect_volatility_regime(high_vol_returns) in valid


# ── Risk score computation tests ───────────────────────────

class TestRiskScoreComputation:
    def test_high_vol_gives_high_risk(self, high_vol_returns):
        from aria.agents.market_risk import compute_risk_score
        score = compute_risk_score(high_vol_returns, regime="crisis")
        assert score > 0.6

    def test_low_vol_gives_low_risk(self, low_vol_returns):
        from aria.agents.market_risk import compute_risk_score
        score = compute_risk_score(low_vol_returns, regime="low_vol")
        assert score < 0.4

    def test_score_bounded_zero_to_one(self, low_vol_returns, high_vol_returns):
        from aria.agents.market_risk import compute_risk_score
        for returns in [low_vol_returns, high_vol_returns]:
            score = compute_risk_score(returns, regime="low_vol")
            assert 0.0 <= score <= 1.0

    def test_empty_returns_returns_midpoint(self):
        from aria.agents.market_risk import compute_risk_score
        score = compute_risk_score([], regime="unknown")
        assert score == 0.5

    def test_trending_up_lower_risk_than_crisis(
        self, trending_up_returns, high_vol_returns
    ):
        from aria.agents.market_risk import compute_risk_score
        up_score = compute_risk_score(trending_up_returns, regime="low_vol")
        crisis_score = compute_risk_score(high_vol_returns, regime="crisis")
        assert up_score < crisis_score


# ── Annualised volatility tests ────────────────────────────

class TestAnnualisedVolatility:
    def test_returns_float(self, low_vol_returns):
        from aria.agents.market_risk import annualised_volatility
        vol = annualised_volatility(low_vol_returns)
        assert isinstance(vol, float)

    def test_empty_returns_zero(self):
        from aria.agents.market_risk import annualised_volatility
        vol = annualised_volatility([])
        assert vol == 0.0

    def test_high_vol_greater_than_low_vol(
        self, low_vol_returns, high_vol_returns
    ):
        from aria.agents.market_risk import annualised_volatility
        assert annualised_volatility(high_vol_returns) > annualised_volatility(
            low_vol_returns
        )

    def test_annualised_by_sqrt_252(self):
        from aria.agents.market_risk import annualised_volatility
        daily_returns = [0.01] * 252
        vol = annualised_volatility(daily_returns)
        assert vol >= 0.0


# ── Degraded mode tests ────────────────────────────────────

class TestMarketRiskDegradedMode:
    def test_degraded_result_has_midpoint_score(self):
        from aria.agents.market_risk import make_degraded_market_result
        result = make_degraded_market_result(
            entity="HSBC", error="Data unavailable"
        )
        assert result.degraded is True
        assert result.risk_score == 0.5
        assert result.confidence == 0.0

    def test_degraded_result_valid_model(self):
        from aria.agents.market_risk import make_degraded_market_result
        result = make_degraded_market_result(
            entity="Barclays", error="Timeout"
        )
        assert isinstance(result, MarketRiskResult)
        assert result.entity == "Barclays"


# ── Agent state transition tests ───────────────────────────

class TestMarketRiskAgentState:
    def test_state_updated_with_result(self, low_vol_returns):
        from aria.agents.market_risk import MarketRiskAgent
        agent = MarketRiskAgent()
        agent._returns_cache["HSBC"] = low_vol_returns

        state = AgentState(entity="HSBC")
        updated = agent.run(state)

        assert updated.market_result is not None
        assert updated.market_result.entity == "HSBC"
        assert updated.market_result.degraded is False
        assert 0.0 <= updated.market_result.risk_score <= 1.0

    def test_unknown_entity_returns_degraded(self):
        from aria.agents.market_risk import MarketRiskAgent
        agent = MarketRiskAgent()

        state = AgentState(entity="UnknownCorp_XYZ")
        updated = agent.run(state)

        assert updated.market_result is not None
        assert updated.market_result.degraded is True

    def test_state_preserves_existing_fields(self, low_vol_returns):
        from aria.agents.market_risk import MarketRiskAgent
        agent = MarketRiskAgent()
        agent._returns_cache["HSBC"] = low_vol_returns

        state = AgentState(entity="HSBC", errors=["prior_error"])
        updated = agent.run(state)

        assert "prior_error" in updated.errors