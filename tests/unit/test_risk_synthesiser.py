"""
Unit tests for the Risk Synthesis Agent.
Tests score combination, risk level routing, and report generation.
"""

import pytest
from datetime import datetime
from aria.config.models import (
    AgentState,
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
def low_risk_state():
    return AgentState(
        entity="HSBC",
        news_result=NewsSentimentResult(
            item_id="n1", entity="HSBC",
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.8, confidence=0.9,
            model_version="v1", latency_ms=10.0,
        ),
        market_result=MarketRiskResult(
            entity="HSBC", timestamp=datetime.utcnow(),
            risk_score=0.15, forecast_return=0.08,
            forecast_volatility=0.10, regime="low_vol",
            confidence=0.85, latency_ms=20.0,
        ),
        counterparty_result=CounterpartyRiskResult(
            entity="HSBC", risk_score=0.20,
            network_centrality=0.3, confidence=0.80,
            latency_ms=15.0,
        ),
        regulatory_result=RegulatoryContext(
            entity="HSBC", compliance_flags=[],
            context_summary="No issues.", confidence=0.95,
            latency_ms=50.0,
        ),
    )


@pytest.fixture
def high_risk_state():
    return AgentState(
        entity="CreditSuisse",
        news_result=NewsSentimentResult(
            item_id="n2", entity="CreditSuisse",
            sentiment_label=SentimentLabel.NEGATIVE,
            sentiment_score=-0.92, confidence=0.95,
            model_version="v1", latency_ms=10.0,
        ),
        market_result=MarketRiskResult(
            entity="CreditSuisse", timestamp=datetime.utcnow(),
            risk_score=0.88, forecast_return=-0.15,
            forecast_volatility=0.45, regime="crisis",
            confidence=0.90, latency_ms=20.0,
        ),
        counterparty_result=CounterpartyRiskResult(
            entity="CreditSuisse", risk_score=0.85,
            network_centrality=0.9,
            contagion_paths=["CreditSuisse -> UBS", "CreditSuisse -> Deutsche Bank"],
            connected_entities=["UBS", "Deutsche Bank"],
            confidence=0.88, latency_ms=15.0,
        ),
        regulatory_result=RegulatoryContext(
            entity="CreditSuisse",
            compliance_flags=["Basel III CET1 breach", "FCA investigation"],
            context_summary="Multiple regulatory concerns.",
            confidence=0.92, latency_ms=50.0,
        ),
    )


@pytest.fixture
def degraded_state():
    """State where all agents returned degraded results."""
    return AgentState(
        entity="UnknownCorp",
        news_result=NewsSentimentResult(
            item_id="n3", entity="UnknownCorp",
            sentiment_label=SentimentLabel.NEUTRAL,
            sentiment_score=0.0, confidence=0.0,
            model_version="v1", latency_ms=0.0,
            degraded=True,
        ),
        market_result=MarketRiskResult(
            entity="UnknownCorp", timestamp=datetime.utcnow(),
            risk_score=0.5, forecast_return=0.0,
            forecast_volatility=0.15, regime="unknown",
            confidence=0.0, latency_ms=0.0,
            degraded=True,
        ),
        counterparty_result=CounterpartyRiskResult(
            entity="UnknownCorp", risk_score=0.5,
            network_centrality=0.5, confidence=0.0,
            latency_ms=0.0, degraded=True,
        ),
        regulatory_result=RegulatoryContext(
            entity="UnknownCorp",
            context_summary="No data.",
            confidence=0.0, latency_ms=0.0,
            degraded=True,
        ),
    )


# ── Risk score computation tests ───────────────────────────

class TestRiskScoreSynthesis:
    def test_low_risk_inputs_give_low_score(self, low_risk_state):
        from aria.agents.risk_synthesiser import compute_aria_score
        score = compute_aria_score(low_risk_state)
        assert score < 40.0

    def test_high_risk_inputs_give_high_score(self, high_risk_state):
        from aria.agents.risk_synthesiser import compute_aria_score
        score = compute_aria_score(high_risk_state)
        assert score > 60.0

    def test_score_bounded_0_to_100(self, low_risk_state, high_risk_state):
        from aria.agents.risk_synthesiser import compute_aria_score
        for state in [low_risk_state, high_risk_state]:
            score = compute_aria_score(state)
            assert 0.0 <= score <= 100.0

    def test_degraded_state_gives_midpoint(self, degraded_state):
        from aria.agents.risk_synthesiser import compute_aria_score
        score = compute_aria_score(degraded_state)
        assert 30.0 <= score <= 70.0

    def test_compliance_flags_increase_score(
        self, low_risk_state, high_risk_state
    ):
        from aria.agents.risk_synthesiser import compute_aria_score
        low_score = compute_aria_score(low_risk_state)
        high_score = compute_aria_score(high_risk_state)
        assert high_score > low_score


# ── Risk level routing tests ───────────────────────────────

class TestRiskLevelRouting:
    def test_score_below_40_is_monitor(self):
        from aria.agents.risk_synthesiser import score_to_risk_level
        assert score_to_risk_level(15.0) == RiskLevel.MONITOR
        assert score_to_risk_level(39.9) == RiskLevel.MONITOR

    def test_score_40_to_60_is_watch(self):
        from aria.agents.risk_synthesiser import score_to_risk_level
        assert score_to_risk_level(40.0) == RiskLevel.WATCH
        assert score_to_risk_level(59.9) == RiskLevel.WATCH

    def test_score_60_to_80_is_alert(self):
        from aria.agents.risk_synthesiser import score_to_risk_level
        assert score_to_risk_level(60.0) == RiskLevel.ALERT
        assert score_to_risk_level(79.9) == RiskLevel.ALERT

    def test_score_80_plus_is_escalate(self):
        from aria.agents.risk_synthesiser import score_to_risk_level
        assert score_to_risk_level(80.0) == RiskLevel.ESCALATE
        assert score_to_risk_level(100.0) == RiskLevel.ESCALATE

    def test_boundary_values(self):
        from aria.agents.risk_synthesiser import score_to_risk_level
        assert score_to_risk_level(0.0) == RiskLevel.MONITOR
        assert score_to_risk_level(100.0) == RiskLevel.ESCALATE


# ── Report generation tests ────────────────────────────────

class TestRiskReportGeneration:
    def test_report_contains_entity(self, high_risk_state):
        from aria.agents.risk_synthesiser import generate_risk_summary
        summary = generate_risk_summary(high_risk_state, score=85.0)
        assert "CreditSuisse" in summary

    def test_report_mentions_risk_level(self, high_risk_state):
        from aria.agents.risk_synthesiser import generate_risk_summary
        summary = generate_risk_summary(high_risk_state, score=85.0)
        assert any(word in summary.lower() for word in
                   ["escalate", "critical", "high", "alert"])

    def test_report_is_string(self, low_risk_state):
        from aria.agents.risk_synthesiser import generate_risk_summary
        summary = generate_risk_summary(low_risk_state, score=20.0)
        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_compliance_flags_in_report(self, high_risk_state):
        from aria.agents.risk_synthesiser import generate_risk_summary
        summary = generate_risk_summary(high_risk_state, score=85.0)
        assert "Basel III" in summary or "FCA" in summary or "regulatory" in summary.lower()


# ── Full agent state transition tests ─────────────────────

class TestRiskSynthesiserAgent:
    def test_low_risk_produces_monitor_event(self, low_risk_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(low_risk_state)
        assert updated.final_risk is not None
        assert updated.final_risk.risk_level == RiskLevel.MONITOR
        assert updated.final_risk.requires_human_review is False

    def test_high_risk_produces_escalate_event(self, high_risk_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(high_risk_state)
        assert updated.final_risk is not None
        assert updated.final_risk.risk_level in [RiskLevel.ALERT, RiskLevel.ESCALATE]

    def test_escalate_requires_human_review(self, high_risk_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(high_risk_state)
        if updated.final_risk.risk_level == RiskLevel.ESCALATE:
            assert updated.final_risk.requires_human_review is True

    def test_final_risk_is_aria_risk_event(self, low_risk_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(low_risk_state)
        assert isinstance(updated.final_risk, ARIARiskEvent)

    def test_degraded_state_still_produces_result(self, degraded_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(degraded_state)
        assert updated.final_risk is not None
        assert updated.final_risk.any_degraded is True

    def test_processing_time_recorded(self, low_risk_state):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        updated = agent.run(low_risk_state)
        assert updated.final_risk.processing_time_ms > 0

    def test_missing_agent_results_handled(self):
        from aria.agents.risk_synthesiser import RiskSynthesiserAgent
        agent = RiskSynthesiserAgent()
        # State with no agent results
        state = AgentState(entity="HSBC")
        updated = agent.run(state)
        assert updated.final_risk is not None
        assert updated.final_risk.any_degraded is True