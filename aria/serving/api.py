"""
ARIA FastAPI Service
Exposes risk assessment endpoints for production use.

Endpoints:
  GET  /health          — liveness check
  POST /assess          — single entity risk assessment
  POST /assess/batch    — batch entity assessment
  GET  /entities        — list of monitored entities
"""

from __future__ import annotations

import uuid
import structlog
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aria.agents.orchestrator import run_aria_pipeline
from aria.config.models import ARIARiskEvent, NewsItem

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="ARIA — Adaptive Risk Intelligence Agent",
    description="Real-time financial risk assessment via multi-agent AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MONITORED_ENTITIES = [
    "HSBC", "Barclays", "Lloyds", "NatWest",
    "Standard Chartered", "Goldman Sachs", "JPMorgan",
    "Deutsche Bank", "Credit Suisse", "UBS",
]


# ── Request/Response models ────────────────────────────────

class AssessRequest(BaseModel):
    entity: str = Field(..., min_length=1, description="Entity to assess")
    headline: Optional[str] = None
    source: Optional[str] = "unknown"


class AssessResponse(BaseModel):
    event_id: str
    entity: str
    aria_risk_score: float
    risk_level: str
    risk_summary: str
    reasoning_trace: str
    confidence: float
    processing_time_ms: float
    any_degraded: bool
    timestamp: datetime


class BatchAssessRequest(BaseModel):
    entities: list[str] = Field(..., min_length=1)


class BatchAssessResponse(BaseModel):
    results: list[AssessResponse]
    total: int
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class EntitiesResponse(BaseModel):
    entities: list[str]
    total: int


# ── Helper ─────────────────────────────────────────────────

def event_to_response(event: ARIARiskEvent) -> AssessResponse:
    return AssessResponse(
        event_id=event.event_id,
        entity=event.entity,
        aria_risk_score=event.aria_risk_score,
        risk_level=event.risk_level.value,
        risk_summary=event.risk_summary,
        reasoning_trace=event.reasoning_trace,
        confidence=event.confidence,
        processing_time_ms=event.processing_time_ms,
        any_degraded=event.any_degraded,
        timestamp=event.timestamp,
    )


# ── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        version="0.1.0",
    )


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest):
    """
    Run ARIA risk assessment for a single entity.
    Optionally include a news headline to trigger news sentiment analysis.
    """
    logger.info("api.assess", entity=request.entity)

    news_item = None
    if request.headline:
        news_item = NewsItem(
            item_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            headline=request.headline,
            source=request.source or "unknown",
            entities=[request.entity],
        )

    try:
        event = run_aria_pipeline(
            entity=request.entity,
            news_item=news_item,
        )
        return event_to_response(event)

    except Exception as e:
        logger.error("api.assess.error", entity=request.entity, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess/batch", response_model=BatchAssessResponse)
def assess_batch(request: BatchAssessRequest):
    """Run ARIA risk assessment for multiple entities."""
    logger.info("api.assess_batch", entities=request.entities)

    results = []
    for entity in request.entities:
        try:
            event = run_aria_pipeline(entity=entity)
            results.append(event_to_response(event))
        except Exception as e:
            logger.error("api.assess_batch.error", entity=entity, error=str(e))

    return BatchAssessResponse(
        results=results,
        total=len(results),
        timestamp=datetime.utcnow(),
    )


@app.get("/entities", response_model=EntitiesResponse)
def list_entities():
    """List all monitored entities."""
    return EntitiesResponse(
        entities=MONITORED_ENTITIES,
        total=len(MONITORED_ENTITIES),
    )