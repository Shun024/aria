"""
ARIA LangGraph Orchestrator
Connects all agents into a directed graph pipeline.

Graph flow:
news_analyst → market_risk → counterparty_risk → regulatory_context → risk_synthesiser

Each node is a pure function: AgentState → AgentState
Errors in any node are caught and logged — pipeline always completes.
"""

from __future__ import annotations

import structlog
from typing import Optional

from langgraph.graph import StateGraph, END

from aria.config.models import AgentState, ARIARiskEvent, NewsItem
from aria.agents.news_analyst import NewsAnalystAgent
from aria.agents.market_risk import MarketRiskAgent
from aria.agents.counterparty_risk import CounterpartyRiskAgent
from aria.agents.regulatory_context import RegulatoryContextAgent
from aria.agents.risk_synthesiser import RiskSynthesiserAgent

logger = structlog.get_logger(__name__)

GRAPH_NODES = [
    "news_analyst",
    "market_risk",
    "counterparty_risk",
    "regulatory_context",
    "risk_synthesiser",
]


def news_analyst_node(state: AgentState) -> AgentState:
    try:
        agent = NewsAnalystAgent()
        return agent.run(state)
    except Exception as e:
        logger.error("orchestrator.news_analyst_node.error", error=str(e))
        return AgentState(
            **{**state.model_dump(),
               "errors": state.errors + [f"news_analyst_node: {e}"]}
        )


def market_risk_node(state: AgentState) -> AgentState:
    try:
        agent = MarketRiskAgent()
        return agent.run(state)
    except Exception as e:
        logger.error("orchestrator.market_risk_node.error", error=str(e))
        return AgentState(
            **{**state.model_dump(),
               "errors": state.errors + [f"market_risk_node: {e}"]}
        )


def counterparty_risk_node(state: AgentState) -> AgentState:
    try:
        agent = CounterpartyRiskAgent()
        return agent.run(state)
    except Exception as e:
        logger.error("orchestrator.counterparty_risk_node.error", error=str(e))
        return AgentState(
            **{**state.model_dump(),
               "errors": state.errors + [f"counterparty_risk_node: {e}"]}
        )


def regulatory_context_node(state: AgentState) -> AgentState:
    try:
        agent = RegulatoryContextAgent()
        return agent.run(state)
    except Exception as e:
        logger.error("orchestrator.regulatory_context_node.error", error=str(e))
        return AgentState(
            **{**state.model_dump(),
               "errors": state.errors + [f"regulatory_context_node: {e}"]}
        )


def risk_synthesiser_node(state: AgentState) -> AgentState:
    agent = RiskSynthesiserAgent()
    return agent.run(state)


def build_aria_graph():
    """Build and compile the ARIA LangGraph pipeline."""
    workflow = StateGraph(AgentState)

    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("market_risk", market_risk_node)
    workflow.add_node("counterparty_risk", counterparty_risk_node)
    workflow.add_node("regulatory_context", regulatory_context_node)
    workflow.add_node("risk_synthesiser", risk_synthesiser_node)

    workflow.set_entry_point("news_analyst")
    workflow.add_edge("news_analyst", "market_risk")
    workflow.add_edge("market_risk", "counterparty_risk")
    workflow.add_edge("counterparty_risk", "regulatory_context")
    workflow.add_edge("regulatory_context", "risk_synthesiser")
    workflow.add_edge("risk_synthesiser", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_aria_graph()
    return _graph


def run_aria_pipeline(
    entity: str,
    news_item: Optional[NewsItem] = None,
) -> ARIARiskEvent:
    """
    Run the full ARIA pipeline for an entity.

    Args:
        entity: financial entity to assess
        news_item: optional news item triggering the assessment

    Returns:
        ARIARiskEvent with unified risk score and reasoning trace
    """
    logger.info("orchestrator.pipeline_start", entity=entity)

    initial_state = AgentState(
        entity=entity,
        news_item=news_item,
    )

    try:
        graph = get_graph()
        final_state = graph.invoke(initial_state)

        if isinstance(final_state, dict):
            final_state = AgentState(**final_state)

        result = final_state.final_risk
        if result is None:
            final_state = RiskSynthesiserAgent().run(final_state)
            result = final_state.final_risk

        logger.info(
            "orchestrator.pipeline_complete",
            entity=entity,
            score=result.aria_risk_score,
            risk_level=result.risk_level.value,
        )
        return result

    except Exception as e:
        logger.error("orchestrator.pipeline_error", entity=entity, error=str(e))
        fallback_state = AgentState(
            entity=entity,
            news_item=news_item,
            errors=[f"pipeline_error: {str(e)}"],
        )
        final_state = RiskSynthesiserAgent().run(fallback_state)
        return final_state.final_risk