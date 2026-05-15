# ADR-003: SQLite for Risk Event Store

**Date:** 2026-05-15  
**Status:** Accepted  
**Author:** Shun Le Yi Mon

## Context

ARIA generates risk events that need to be persisted for audit trail,
dashboard queries, and drift monitoring.

## Decision

Use SQLite via SQLAlchemy async for the risk event store.

## Reasoning

- **Simplicity** — zero infrastructure, file-based, portable
- **Sufficient** — ARIA generates ~100-1000 events/day, well within SQLite limits
- **Async support** — aiosqlite enables non-blocking I/O in FastAPI
- **Auditable** — SQLite files are portable and inspectable

## When to migrate

Migrate to PostgreSQL when:
- Event volume exceeds 100k/day
- Multiple writer processes needed
- Full-text search on risk summaries required

## Consequences

- Single writer at a time (acceptable for current throughput)
- DB file backed up daily via script
- SQLAlchemy ORM enables future migration with minimal code change