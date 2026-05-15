"""
Unit tests for stub agents (counterparty risk + regulatory context).
These will be replaced with full GNN and RAG implementations in v2.
"""

import pytest
from aria.config.models import AgentState, CounterpartyRiskResult, RegulatoryContext


class TestCounterpartyRiskAgent:
    def test_known_entity_returns_profile(self):
        from aria.agents.counterparty_risk import CounterpartyRiskAgent
        agent = CounterpartyRiskAgent()
        state = AgentState(entity="HSBC")
        updated = agent.run(state)
        assert updated.counterparty_result is not None
        assert updated.counterparty_result.entity == "HSBC"
        assert updated.counterparty_result.degraded is False
        assert updated.counterparty_result.risk_score == 0.25

    def test_unknown_entity_returns_degraded(self):
        from aria.agents.counterparty_risk import CounterpartyRiskAgent
        agent = CounterpartyRiskAgent()
        state = AgentState(entity="UnknownCorp")
        updated = agent.run(state)
        assert updated.counterparty_result.degraded is True
        assert updated.counterparty_result.confidence == 0.0

    def test_high_risk_entity_scores_correctly(self):
        from aria.agents.counterparty_risk import CounterpartyRiskAgent
        agent = CounterpartyRiskAgent()
        state = AgentState(entity="Credit Suisse")
        updated = agent.run(state)
        assert updated.counterparty_result.risk_score > 0.5

    def test_result_is_valid_model(self):
        from aria.agents.counterparty_risk import CounterpartyRiskAgent
        agent = CounterpartyRiskAgent()
        state = AgentState(entity="Barclays")
        updated = agent.run(state)
        assert isinstance(updated.counterparty_result, CounterpartyRiskResult)

    def test_connected_entities_populated(self):
        from aria.agents.counterparty_risk import CounterpartyRiskAgent
        agent = CounterpartyRiskAgent()
        state = AgentState(entity="HSBC")
        updated = agent.run(state)
        assert len(updated.counterparty_result.connected_entities) > 0


class TestRegulatoryContextAgent:
    def test_known_entity_returns_profile(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="HSBC")
        updated = agent.run(state)
        assert updated.regulatory_result is not None
        assert updated.regulatory_result.entity == "HSBC"
        assert updated.regulatory_result.degraded is False

    def test_unknown_entity_returns_degraded(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="UnknownCorp")
        updated = agent.run(state)
        assert updated.regulatory_result.degraded is True
        assert updated.regulatory_result.confidence == 0.0

    def test_high_risk_entity_has_flags(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="Credit Suisse")
        updated = agent.run(state)
        assert len(updated.regulatory_result.compliance_flags) > 0

    def test_clean_entity_has_no_flags(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="HSBC")
        updated = agent.run(state)
        assert updated.regulatory_result.compliance_flags == []

    def test_result_is_valid_model(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="Barclays")
        updated = agent.run(state)
        assert isinstance(updated.regulatory_result, RegulatoryContext)

    def test_context_summary_not_empty(self):
        from aria.agents.regulatory_context import RegulatoryContextAgent
        agent = RegulatoryContextAgent()
        state = AgentState(entity="Deutsche Bank")
        updated = agent.run(state)
        assert len(updated.regulatory_result.context_summary) > 10