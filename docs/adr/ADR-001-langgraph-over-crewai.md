# ADR-001: LangGraph over CrewAI for Agent Orchestration

**Date:** 2026-05-15  
**Status:** Accepted  
**Author:** Shun Le Yi Mon

## Context

ARIA requires orchestrating 4 specialised agents (news, market, counterparty, regulatory) 
whose outputs feed into a synthesis agent. We evaluated LangGraph and CrewAI.

## Decision

Use LangGraph.

## Reasoning

| Criterion | LangGraph | CrewAI |
|---|---|---|
| State management | Explicit typed state (AgentState) | Implicit, harder to test |
| Testability | Each node is a pure function | Agents are harder to unit test |
| Conditional routing | First-class (edges + conditions) | Limited |
| Observability | Full trace via LangSmith | Limited |
| Production maturity | Used at scale in production | Earlier stage |
| Graceful degradation | Easy — check state at each node | Harder to implement |

## Consequences

- Each agent is a pure function: `AgentState -> AgentState`
- State is fully typed via Pydantic — no raw dicts
- Easy to unit test each agent in isolation
- Conditional routing enables human-in-the-loop escalation naturally