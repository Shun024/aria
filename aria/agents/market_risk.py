"""
ARIA Market Risk Agent
Computes market risk score from historical returns data.

Uses statistical measures (annualised volatility, regime detection)
rather than a live ML model — fast, interpretable, no external deps.
In production this would call the TFT model from MarketPulse.
"""

from __future__ import annotations

import time
import math
import structlog
from datetime import datetime

import numpy as np

from aria.config.models import AgentState, MarketRiskResult

logger = structlog.get_logger(__name__)

MODEL_VERSION = "market-risk-v1.0"

# Volatility thresholds (annualised)
VOL_LOW_THRESHOLD = 0.12    # <12% annualised vol = low
VOL_HIGH_THRESHOLD = 0.25   # >25% annualised vol = high
VOL_CRISIS_THRESHOLD = 0.40 # >40% annualised vol = crisis

# Synthetic returns for known entities (in production: from yfinance/Kafka)
ENTITY_RETURNS_SEED = {
    "HSBC": (0.0003, 0.012),         # mean, std of daily returns
    "Barclays": (0.0002, 0.015),
    "Lloyds": (0.0004, 0.011),
    "NatWest": (0.0003, 0.013),
    "Standard Chartered": (0.0001, 0.016),
    "Goldman Sachs": (0.0005, 0.014),
    "JPMorgan": (0.0006, 0.013),
    "Deutsche Bank": (0.0001, 0.018),
    "Credit Suisse": (-0.0002, 0.022),
}


# ── Pure helper functions ──────────────────────────────────

def annualised_volatility(returns: list[float]) -> float:
    """
    Compute annualised volatility from daily returns.

    Args:
        returns: list of daily return values

    Returns:
        Annualised volatility (0.0 if insufficient data)
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    daily_vol = float(np.std(arr, ddof=1))
    return round(daily_vol * math.sqrt(252), 4)


def detect_volatility_regime(returns: list[float]) -> str:
    """
    Classify current volatility regime.

    Args:
        returns: list of daily return values

    Returns:
        One of: "low_vol", "high_vol", "crisis", "unknown"
    """
    if not returns:
        return "unknown"

    vol = annualised_volatility(returns)

    if vol < VOL_LOW_THRESHOLD:
        return "low_vol"
    elif vol < VOL_HIGH_THRESHOLD:
        return "high_vol"
    elif vol >= VOL_CRISIS_THRESHOLD:
        return "crisis"
    else:
        return "high_vol"


def compute_risk_score(returns: list[float], regime: str) -> float:
    """
    Compute normalised risk score [0, 1] from returns and regime.

    Args:
        returns: list of daily return values
        regime: volatility regime string

    Returns:
        Risk score in [0, 1]
    """
    if not returns:
        return 0.5

    arr = np.array(returns)
    vol = annualised_volatility(returns)

    # Normalise volatility to [0, 1] using crisis threshold as ceiling
    vol_score = min(vol / VOL_CRISIS_THRESHOLD, 1.0)

    # Drawdown component
    cumulative = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = float(np.min((cumulative - peak) / peak))
    drawdown_score = min(abs(drawdown) / 0.3, 1.0)  # 30% drawdown = max risk

    # Regime multiplier
    regime_multiplier = {
        "low_vol": 0.7,
        "high_vol": 1.1,
        "crisis": 1.4,
        "unknown": 1.0,
    }.get(regime, 1.0)

    raw_score = (vol_score * 0.6 + drawdown_score * 0.4) * regime_multiplier
    return round(max(0.0, min(1.0, raw_score)), 4)


def make_degraded_market_result(entity: str, error: str) -> MarketRiskResult:
    """Return neutral degraded result when market data unavailable."""
    logger.warning(
        "market_risk.degraded",
        entity=entity,
        error=error,
    )
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


# ── Agent class ────────────────────────────────────────────

class MarketRiskAgent:
    """
    Market Risk Agent — computes statistical market risk score.

    Uses synthetic returns seeded from known entity profiles.
    In production: pulls from yfinance or Kafka market data stream.
    Gracefully degrades for unknown entities.
    """

    def __init__(self, n_days: int = 252):
        self.n_days = n_days
        self._returns_cache: dict[str, list[float]] = {}
        self._rng = np.random.default_rng(seed=42)

    def _get_returns(self, entity: str) -> list[float] | None:
        """Get or generate returns for entity."""
        if entity in self._returns_cache:
            return self._returns_cache[entity]

        if entity in ENTITY_RETURNS_SEED:
            mean, std = ENTITY_RETURNS_SEED[entity]
            returns = list(
                self._rng.normal(mean, std, self.n_days).astype(float)
            )
            self._returns_cache[entity] = returns
            return returns

        return None

    def run(self, state: AgentState) -> AgentState:
        """
        Compute market risk for entity in state.
        AgentState -> AgentState (with market_result populated)
        """
        t0 = time.time()
        entity = state.entity

        try:
            returns = self._get_returns(entity)

            if returns is None:
                logger.warning(
                    "market_risk.unknown_entity",
                    entity=entity,
                )
                result = make_degraded_market_result(
                    entity=entity,
                    error=f"No market data for {entity}",
                )
                return AgentState(**{
                    **state.model_dump(),
                    "market_result": result,
                    "errors": state.errors + [
                        f"market_risk: no data for {entity}"
                    ],
                })

            regime = detect_volatility_regime(returns)
            risk_score = compute_risk_score(returns, regime)
            vol = annualised_volatility(returns)
            arr = np.array(returns)
            forecast_return = float(np.mean(arr) * 252)
            latency_ms = (time.time() - t0) * 1000

            result = MarketRiskResult(
                entity=entity,
                timestamp=datetime.utcnow(),
                risk_score=risk_score,
                forecast_return=round(forecast_return, 4),
                forecast_volatility=vol,
                regime=regime,
                confidence=0.75,
                latency_ms=round(latency_ms, 2),
                degraded=False,
            )

            logger.info(
                "market_risk.scored",
                entity=entity,
                risk_score=risk_score,
                regime=regime,
                vol=vol,
                latency_ms=round(latency_ms, 2),
            )

            return AgentState(**{**state.model_dump(), "market_result": result})

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(
                "market_risk.error",
                entity=entity,
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            result = make_degraded_market_result(entity=entity, error=str(e))
            return AgentState(**{
                **state.model_dump(),
                "market_result": result,
                "errors": state.errors + [f"market_risk: {str(e)}"],
            })