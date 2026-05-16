# 🛡️ ARIA — Adaptive Risk Intelligence Agent

A production-grade multi-agent AI system for real-time financial risk assessment. Five specialised agents orchestrated via LangGraph analyse news sentiment, market risk, counterparty exposure, and regulatory context — synthesising a unified ARIA Risk Score with full reasoning trace.

**Built to staff/senior engineer standards:** 105 unit tests, 93% coverage, typed Pydantic models throughout, CI/CD pipeline, structured logging, graceful degradation, and architecture decision records.

---

## Architecture

```
News Stream (Kafka) + Manual Headlines
        │
        ▼
┌─────────────────────────────────────────┐
│         LangGraph Orchestrator          │
│                                         │
│  NewsAnalyst → MarketRisk →             │
│  CounterpartyRisk → RegulatoryContext → │
│  RiskSynthesiser                        │
└─────────────────────────────────────────┘
        │
        ▼
ARIA Risk Score [0–100] + Risk Level + Reasoning Trace
        │
        ├── FastAPI REST API
        └── Plotly Dash Dashboard
```

---

## Agents

| Agent | Model | Signal |
|---|---|---|
| **News Analyst** | FinBERT (ProsusAI) | Sentiment score [-1, 1] from financial headlines |
| **Market Risk** | Statistical (TFT in v2) | Volatility regime + drawdown risk [0, 1] |
| **Counterparty Risk** | Network profiles (GNN in v2) | Network centrality + contagion paths |
| **Regulatory Context** | Rule-based (RAG in v2) | Compliance flags + Basel III / FCA context |
| **Risk Synthesiser** | Weighted combination | Unified ARIA Score + risk level routing |

---

## Risk Levels

| Level | Score | Action |
|---|---|---|
| 🟢 MONITOR | 0–40 | Passive observation |
| 🟡 WATCH | 40–60 | Increased attention |
| 🟠 ALERT | 60–80 | Analyst notification |
| 🔴 ESCALATE | 80–100 | Immediate human review |

---

## Engineering Standards

- **105 unit tests** across models, agents, orchestrator, and API — 93.37% coverage
- **Pydantic v2** typed models for all inter-agent data — no raw dicts
- **Graceful degradation** — every agent returns a neutral result on failure, system never crashes
- **Structured logging** via structlog — every agent call logged with entity, latency, score
- **CI/CD** via GitHub Actions — lint (ruff) + type check (mypy) + tests on every push
- **Architecture Decision Records** in `docs/adr/` — LangGraph, ChromaDB, SQLite choices documented
- **Configuration** via Pydantic settings — no hardcoded values

---

## Sample Output

```
Entity: Credit Suisse
ARIA Score: 94.2 / 100
Risk Level: ESCALATE 🔴
Requires Human Review: True

Reasoning: Score=94.2 | News=-0.961 (w=0.3) | Market=0.980 (w=0.4) |
           Counterparty=0.800 (w=0.3) | RegFlags=2 | Degraded=False

Regulatory flags: Basel III CET1 breach, FINMA investigation
```

---

## Stack

LangGraph · FinBERT · FastAPI · Plotly Dash · Pydantic v2 · structlog · MLflow · pytest · GitHub Actions · Docker

---

## Quickstart

```bash
git clone https://github.com/Shun024/aria.git
cd aria
python3.11 -m venv .venv && source .venv/bin/activate
pip install pydantic pydantic-settings structlog numpy pandas scikit-learn \
            langgraph langchain langchain-openai "transformers==4.44.0" \
            torch fastapi uvicorn httpx dash dash-bootstrap-components plotly

cp .env.example .env
# Add OPENAI_API_KEY and GROQ_API_KEY

# Run tests
PYTHONPATH=. pytest tests/unit/ -v

# Run pipeline
PYTHONPATH=. python -c "
from aria.agents.orchestrator import run_aria_pipeline
r = run_aria_pipeline('Credit Suisse')
print(f'Score: {r.aria_risk_score} | Level: {r.risk_level.value}')
"

# Launch dashboard
PYTHONPATH=. python aria/dashboard/app.py
# Open http://localhost:8056

# Launch API
PYTHONPATH=. uvicorn aria.serving.api:app --port 8010
# Docs at http://localhost:8010/docs
```

---

## Roadmap (v2)

- [ ] Full GNN counterparty risk (replacing stub profiles)
- [ ] RAG regulatory context over Basel III / FCA documents
- [ ] Kafka news stream integration
- [ ] TFT market risk forecasting (from MarketPulse)
- [ ] Human-in-the-loop escalation UI
- [ ] Evidently drift monitoring
- [ ] PostgreSQL for production event store

---

## Author

**Shun Le Yi Mon (Sheryl)** · Data Scientist · NLP & GenAI  
[LinkedIn](https://linkedin.com/in/shunleyimon724) · [GitHub](https://github.com/Shun024)