# ADR-002: ChromaDB for Vector Store

**Date:** 2026-05-15  
**Status:** Accepted  
**Author:** Shun Le Yi Mon

## Context

ARIA's regulatory RAG agent needs a vector store for Basel III, FCA, 
and PRA regulatory documents.

## Decision

Use ChromaDB (local persistent mode for development, server mode for production).

## Reasoning

- **No infrastructure cost** — runs locally, no managed service needed
- **Python-native** — simple integration, no separate service for dev
- **Proven** — used in FinSight Analyst with good results
- **Upgradeable** — can switch to Pinecone/Weaviate with minimal code change
  via abstraction layer

## Consequences

- Regulatory documents ingested once, persisted to disk
- Collection rebuilt on schema changes via migration script
- Production deployment uses ChromaDB server mode (Docker)