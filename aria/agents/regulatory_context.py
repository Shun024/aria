"""
ARIA Regulatory Context Agent (stub — full RAG implementation in v2).
Returns relevant regulatory context for known entities.
"""

from __future__ import annotations
import time
import structlog
from aria.config.models import AgentState, RegulatoryContext

logger = structlog.get_logger(__name__)

REGULATORY_PROFILES = {
    "HSBC": {
        "regulations": ["Basel III", "FCA SYSC", "PRA Rulebook"],
        "flags": [],
        "summary": "HSBC meets Basel III CET1 requirements. No active FCA investigations.",
    },
    "Barclays": {
        "regulations": ["Basel III", "FCA SYSC"],
        "flags": [],
        "summary": "Barclays meets capital requirements. Minor PRA correspondence ongoing.",
    },
    "Credit Suisse": {
        "regulations": ["Basel III", "FINMA", "FCA"],
        "flags": ["Basel III CET1 breach", "FINMA investigation"],
        "summary": "Credit Suisse faces multiple regulatory challenges including capital concerns.",
    },
    "Deutsche Bank": {
        "regulations": ["Basel III", "BaFin", "FCA"],
        "flags": ["AML compliance review"],
        "summary": "Deutsche Bank under AML review. Capital ratios within acceptable range.",
    },
}


class RegulatoryContextAgent:
    def run(self, state: AgentState) -> AgentState:
        t0 = time.time()
        entity = state.entity
        profile = REGULATORY_PROFILES.get(entity)

        if profile is None:
            result = RegulatoryContext(
                entity=entity,
                context_summary="No regulatory profile available.",
                confidence=0.0,
                latency_ms=0.0,
                degraded=True,
            )
        else:
            result = RegulatoryContext(
                entity=entity,
                relevant_regulations=profile["regulations"],
                compliance_flags=profile["flags"],
                context_summary=profile["summary"],
                confidence=0.85,
                latency_ms=round((time.time() - t0) * 1000, 2),
            )

        return AgentState(**{**state.model_dump(), "regulatory_result": result})