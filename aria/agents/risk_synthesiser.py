"""
ARIA Risk Synthesiser Agent
Combines signals from all agents into a unified ARIA Risk Score.

Weighted combination:
- News sentiment: 30% (immediate market reaction)
- Market risk: 40% (quantitative risk signal)
- Counterparty risk: 30% (network contagion risk)
- Regulatory flags: additive bonus (non-negotiable risk factors)

Risk levels:
- MONITOR (0-40): passive observation
- WATCH (40-60): increased attention
- ALERT (60-80): analyst notification
- ESCALATE (80-100): immediate human review
"""

from __future__ import annotations

import time
import uuid
import structlog
from datetime import datetime

from aria.config.models import (
    AgentState,
    ARIARiskEvent,
    CounterpartyRiskResult,
    MarketRiskResult,
    NewsSentimentResult,
    RegulatoryContext,
    RiskLevel,
    SentimentLabel,
)

logger = structlog.get_logger(__name__)

# Signal weights — must sum to 1.0
NEWS_WEIGHT = 0.30
MARKET_WEIGHT = 0.40
COUNTERPARTY_WEIGHT = 0.30

# Risk level thresholds
WATCH_THRESHOLD = 40.0
ALERT_THRESHOLD = 60.0
ESCALATE_THRESHOLD = 80.0

# Regulatory flag penalty per flag (additive)
REGULATORY_FLAG_PENALTY = 8.0
MAX_REGULATORY_PENALTY = 25.0


def _default_news_result(entity: str) -> NewsSentimentResult:
    return NewsSentimentResult(
        item_id="default",
        entity=entity,
        sentiment_label=SentimentLabel.NEUTRAL,
        sentiment_score=0.0,
        confidence=0.0,
        model_version="default",
        latency_ms=0.0,
        degraded=True,
    )


def _default_market_result(entity: str) -> MarketRiskResult:
    return MarketRiskResult(
        entity=entity,
        timestamp=datetime.utcnow(),
        risk_score=0.5,
        forecast_return=0.0,
        forecast_volatility=0.15,
        regime="unknown",
        confidence=0.0,
        latency_ms=0.0,
        degraded=True,
    )


def _default_counterparty_result(entity: str) -> CounterpartyRiskResult:
    return CounterpartyRiskResult(
        entity=entity,
        risk_score=0.5,
        network_centrality=0.5,
        confidence=0.0,
        latency_ms=0.0,
        degraded=True,
    )


def _default_regulatory_result(entity: str) -> RegulatoryContext:
    return RegulatoryContext(
        entity=entity,
        context_summary="No regulatory data available.",
        confidence=0.0,
        latency_ms=0.0,
        degraded=True,
    )


def compute_aria_score(state: AgentState) -> float:
    """
    Compute unified ARIA Risk Score [0, 100].

    Combines news sentiment, market risk, and counterparty risk
    with configurable weights, plus additive regulatory penalty.

    Args:
        state: AgentState with all agent results populated

    Returns:
        ARIA Risk Score in [0, 100]
    """
    news = state.news_result or _default_news_result(state.entity)
    market = state.market_result or _default_market_result(state.entity)
    counterparty = state.counterparty_result or _default_counterparty_result(
        state.entity
    )
    regulatory = state.regulatory_result or _default_regulatory_result(
        state.entity
    )

    # Convert news sentiment [-1, 1] to risk [0, 1]
    # Negative sentiment → high risk
    news_risk = (1.0 - news.sentiment_score) / 2.0
    news_risk = max(0.0, min(1.0, news_risk))

    # Weighted combination
    base_score = (
        NEWS_WEIGHT * news_risk
        + MARKET_WEIGHT * market.risk_score
        + COUNTERPARTY_WEIGHT * counterparty.risk_score
    ) * 100.0

    # Regulatory penalty — additive, capped
    n_flags = len(regulatory.compliance_flags)
    regulatory_penalty = min(
        n_flags * REGULATORY_FLAG_PENALTY,
        MAX_REGULATORY_PENALTY,
    )

    final_score = base_score + regulatory_penalty
    return round(max(0.0, min(100.0, final_score)), 2)


def score_to_risk_level(score: float) -> RiskLevel:
    """
    Map ARIA score to risk level.

    Args:
        score: ARIA Risk Score [0, 100]

    Returns:
        RiskLevel enum value
    """
    if score >= ESCALATE_THRESHOLD:
        return RiskLevel.ESCALATE
    elif score >= ALERT_THRESHOLD:
        return RiskLevel.ALERT
    elif score >= WATCH_THRESHOLD:
        return RiskLevel.WATCH
    else:
        return RiskLevel.MONITOR


def generate_risk_summary(state: AgentState, score: float) -> str:
    """
    Generate natural language risk summary.

    Args:
        state: AgentState with all results
        score: computed ARIA score

    Returns:
        Human-readable risk summary string
    """
    entity = state.entity
    risk_level = score_to_risk_level(score)
    news = state.news_result
    market = state.market_result
    counterparty = state.counterparty_result
    regulatory = state.regulatory_result

    lines = [
        f"ARIA Risk Assessment for {entity} — Score: {score:.1f}/100 [{risk_level.value}]",
    ]

    if news and not news.degraded:
        sentiment_str = news.sentiment_label.value.upper()
        lines.append(
            f"News sentiment: {sentiment_str} "
            f"(score={news.sentiment_score:+.2f}, confidence={news.confidence:.0%})"
        )

    if market and not market.degraded:
        lines.append(
            f"Market risk: {market.risk_score:.2f} "
            f"[regime={market.regime}, vol={market.forecast_volatility:.0%}]"
        )

    if counterparty and not counterparty.degraded:
        lines.append(
            f"Counterparty risk: {counterparty.risk_score:.2f} "
            f"[centrality={counterparty.network_centrality:.2f}]"
        )
        if counterparty.contagion_paths:
            lines.append(
                f"Contagion paths: {', '.join(counterparty.contagion_paths[:2])}"
            )

    if regulatory and regulatory.compliance_flags:
        flags_str = ", ".join(regulatory.compliance_flags)
        lines.append(f"Regulatory flags: {flags_str}")

    if risk_level == RiskLevel.ESCALATE:
        lines.append(
            "⚠️  ESCALATE: Immediate human review required. "
            "Multiple high-risk signals detected."
        )
    elif risk_level == RiskLevel.ALERT:
        lines.append(
            "🔔 ALERT: Analyst notification triggered. "
            "Elevated risk signals require attention."
        )
    elif risk_level == RiskLevel.WATCH:
        lines.append(
            "👁️  WATCH: Increased monitoring activated."
        )

    return " | ".join(lines)


class RiskSynthesiserAgent:
    """
    Risk Synthesiser Agent — final orchestration step.

    Combines all agent signals into a unified ARIARiskEvent.
    Always produces a result — degrades gracefully if agents failed.
    """

    def run(self, state: AgentState) -> AgentState:
        """
        Synthesise all agent results into final risk event.
        Always succeeds — worst case returns degraded event.
        """
        t0 = time.time()
        entity = state.entity

        try:
            # Fill defaults for any missing agent results
            news = state.news_result or _default_news_result(entity)
            market = state.market_result or _default_market_result(entity)
            counterparty = (
                state.counterparty_result
                or _default_counterparty_result(entity)
            )
            regulatory = (
                state.regulatory_result or _default_regulatory_result(entity)
            )

            any_degraded = any([
                news.degraded,
                market.degraded,
                counterparty.degraded,
                regulatory.degraded,
            ])

            score = compute_aria_score(state)
            risk_level = score_to_risk_level(score)
            summary = generate_risk_summary(state, score)
            processing_time_ms = (time.time() - t0) * 1000

            # Build reasoning trace
            reasoning = (
                f"Score={score:.1f} | "
                f"News={abs(news.sentiment_score):.2f} (w={NEWS_WEIGHT}) | "
                f"Market={market.risk_score:.2f} (w={MARKET_WEIGHT}) | "
                f"Counterparty={counterparty.risk_score:.2f} (w={COUNTERPARTY_WEIGHT}) | "
                f"RegFlags={len(regulatory.compliance_flags)} | "
                f"Degraded={any_degraded}"
            )

            event = ARIARiskEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                entity=entity,
                news_sentiment=news,
                market_risk=market,
                counterparty_risk=counterparty,
                regulatory_context=regulatory,
                aria_risk_score=score,
                risk_level=risk_level,
                risk_summary=summary,
                reasoning_trace=reasoning,
                confidence=self._compute_confidence(
                    news, market, counterparty, regulatory
                ),
                processing_time_ms=round(processing_time_ms, 2),
                any_degraded=any_degraded,
            )

            logger.info(
                "risk_synthesiser.complete",
                entity=entity,
                score=score,
                risk_level=risk_level.value,
                requires_human_review=event.requires_human_review,
                processing_time_ms=round(processing_time_ms, 2),
            )

            return AgentState(**{**state.model_dump(), "final_risk": event})

        except Exception as e:
            logger.error(
                "risk_synthesiser.error",
                entity=entity,
                error=str(e),
            )
            # Even on failure, return a degraded event
            processing_time_ms = (time.time() - t0) * 1000
            event = ARIARiskEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                entity=entity,
                news_sentiment=_default_news_result(entity),
                market_risk=_default_market_result(entity),
                counterparty_risk=_default_counterparty_result(entity),
                regulatory_context=_default_regulatory_result(entity),
                aria_risk_score=50.0,
                risk_level=RiskLevel.WATCH,
                risk_summary=f"System error during synthesis: {str(e)}",
                reasoning_trace=f"Error: {str(e)}",
                confidence=0.0,
                processing_time_ms=round(processing_time_ms, 2),
                any_degraded=True,
            )
            return AgentState(**{
                **state.model_dump(),
                "final_risk": event,
                "errors": state.errors + [f"risk_synthesiser: {str(e)}"],
            })

    @staticmethod
    def _compute_confidence(
        news: NewsSentimentResult,
        market: MarketRiskResult,
        counterparty: CounterpartyRiskResult,
        regulatory: RegulatoryContext,
    ) -> float:
        """Average confidence across non-degraded agents."""
        confidences = [
            c for agent, c in [
                (news, news.confidence),
                (market, market.confidence),
                (counterparty, counterparty.confidence),
                (regulatory, regulatory.confidence),
            ]
            if not agent.degraded
        ]
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 4)