"""
ARIA — Adaptive Risk Intelligence Agent Dashboard
Real-time financial risk monitoring interface.
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from aria.agents.orchestrator import run_aria_pipeline
from aria.config.models import ARIARiskEvent, NewsItem, RiskLevel

# ── Entities to monitor ────────────────────────────────────
MONITORED_ENTITIES = [
    "HSBC", "Barclays", "Lloyds", "NatWest",
    "Deutsche Bank", "Credit Suisse",
]

RISK_COLOURS = {
    "MONITOR": "#00FF7F",
    "WATCH":   "#FFD700",
    "ALERT":   "#FF8C00",
    "ESCALATE": "#FF4444",
}

ENTITY_COLOURS = {
    "HSBC": "#4488FF",
    "Barclays": "#7B68EE",
    "Lloyds": "#00CC88",
    "NatWest": "#FFB300",
    "Deutsche Bank": "#FF8844",
    "Credit Suisse": "#FF4444",
}

# ── In-memory state ────────────────────────────────────────
risk_cache: dict[str, ARIARiskEvent] = {}
risk_history: dict[str, list[dict]] = {e: [] for e in MONITORED_ENTITIES}


def run_assessment(entity: str, headline: Optional[str] = None) -> ARIARiskEvent:
    """Run pipeline and cache result."""
    news_item = None
    if headline:
        news_item = NewsItem(
            item_id=f"dash_{entity}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            headline=headline,
            source="manual",
            entities=[entity],
        )
    result = run_aria_pipeline(entity=entity, news_item=news_item)
    risk_cache[entity] = result
    risk_history[entity].append({
        "timestamp": datetime.utcnow().isoformat(),
        "score": result.aria_risk_score,
        "risk_level": result.risk_level.value,
    })
    if len(risk_history[entity]) > 50:
        risk_history[entity] = risk_history[entity][-50:]
    return result


def initialise_cache():
    """Pre-populate cache on startup."""
    for entity in MONITORED_ENTITIES:
        try:
            run_assessment(entity)
        except Exception:
            pass


# Pre-populate in background
threading.Thread(target=initialise_cache, daemon=True).start()

# ── App ────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="ARIA — Risk Intelligence",
)


def risk_badge(level: str) -> html.Span:
    colour = RISK_COLOURS.get(level, "#888")
    return html.Span(
        level,
        style={
            "backgroundColor": colour,
            "color": "#000",
            "padding": "2px 10px",
            "borderRadius": "4px",
            "fontWeight": "bold",
            "fontSize": "0.75rem",
        },
    )


def score_gauge(score: float, entity: str) -> go.Figure:
    colour = RISK_COLOURS[
        "ESCALATE" if score >= 80 else
        "ALERT" if score >= 60 else
        "WATCH" if score >= 40 else "MONITOR"
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": entity, "font": {"size": 12}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 40], "color": "rgba(0,255,127,0.1)"},
                {"range": [40, 60], "color": "rgba(255,215,0,0.1)"},
                {"range": [60, 80], "color": "rgba(255,140,0,0.1)"},
                {"range": [80, 100], "color": "rgba(255,68,68,0.1)"},
            ],
            "threshold": {
                "line": {"color": colour, "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"font": {"color": colour, "size": 28}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🛡️ ARIA", className="mt-3 mb-0",
                    style={"color": "#00FF7F"}),
            html.P(
                "Adaptive Risk Intelligence Agent · "
                "Real-Time Financial Risk Monitoring",
                className="text-muted mb-0",
                style={"fontSize": "0.85rem"},
            ),
        ], md=8),
        dbc.Col([
            html.Div(id="last-updated", className="text-muted text-end mt-3",
                     style={"fontSize": "0.75rem"}),
        ], md=4),
    ]),

    html.Hr(),

    # Manual assessment input
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Select(
                    id="entity-select",
                    options=[{"label": e, "value": e} for e in MONITORED_ENTITIES],
                    value="HSBC",
                ),
                dbc.Input(
                    id="headline-input",
                    placeholder="Optional: paste a news headline to include sentiment analysis...",
                    type="text",
                ),
                dbc.Button(
                    "Assess", id="assess-btn", color="success", n_clicks=0
                ),
            ]),
        ])
    ], className="mb-4"),

    # Gauge charts
    dbc.Row([
        dbc.Col([
            dcc.Graph(id=f"gauge-{entity}", figure=score_gauge(50, entity),
                      style={"height": "180px"})
        ], md=2)
        for entity in MONITORED_ENTITIES
    ]),

    html.Hr(),

    # Detail panel
    dbc.Row([
        dbc.Col([
            html.H5("Risk Assessment Detail", className="mb-3"),
            html.Div(id="detail-panel"),
        ], md=6),
        dbc.Col([
            html.H5("Score History", className="mb-3"),
            dcc.Graph(id="history-chart", style={"height": "300px"}),
        ], md=6),
    ]),

    html.Hr(),

    # Risk summary table
    dbc.Row([
        dbc.Col([
            html.H5("Entity Risk Summary", className="mb-3"),
            html.Div(id="summary-table"),
        ])
    ]),

    # Refresh interval
    dcc.Interval(id="refresh", interval=30_000, n_intervals=0),
    dcc.Store(id="selected-entity", data="HSBC"),

    html.Hr(className="mt-4"),
    html.P(
        "ARIA v0.1.0 · Agents: FinBERT News Analyst · "
        "Market Risk · Counterparty Risk · Regulatory Context · "
        "LangGraph Orchestration",
        className="text-muted text-center mb-3",
        style={"fontSize": "0.7rem"},
    ),

], fluid=True)


@app.callback(
    [Output("selected-entity", "data")] +
    [Output(f"gauge-{e}", "figure") for e in MONITORED_ENTITIES] +
    [Output("detail-panel", "children"),
     Output("history-chart", "figure"),
     Output("summary-table", "children"),
     Output("last-updated", "children")],
    [Input("assess-btn", "n_clicks"),
     Input("refresh", "n_intervals")],
    [State("entity-select", "value"),
     State("headline-input", "value"),
     State("selected-entity", "data")],
)
def update_dashboard(n_clicks, n_intervals, entity, headline, selected):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Run assessment on button click
    if "assess-btn" in triggered and entity:
        try:
            run_assessment(entity, headline or None)
            selected = entity
        except Exception:
            pass

    # Build gauges
    gauges = []
    for e in MONITORED_ENTITIES:
        result = risk_cache.get(e)
        score = result.aria_risk_score if result else 50.0
        gauges.append(score_gauge(score, e))

    # Detail panel for selected entity
    result = risk_cache.get(selected)
    if result:
        colour = RISK_COLOURS.get(result.risk_level.value, "#888")
        detail = [
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H4(f"{selected}", className="mb-1"),
                        risk_badge(result.risk_level.value),
                        html.Span(
                            f"  Score: {result.aria_risk_score:.1f}/100",
                            style={"color": colour, "fontWeight": "bold",
                                   "marginLeft": "10px"},
                        ),
                    ]),
                    html.Hr(),
                    html.P(result.risk_summary,
                           style={"fontSize": "0.8rem", "color": "#ccc"}),
                    html.Hr(),
                    html.H6("Reasoning Trace", className="text-muted"),
                    html.P(result.reasoning_trace,
                           style={"fontSize": "0.75rem",
                                  "fontFamily": "monospace",
                                  "color": "#aaa"}),
                    html.Small(
                        f"Processing time: {result.processing_time_ms:.0f}ms | "
                        f"Confidence: {result.confidence:.0%} | "
                        f"Degraded: {result.any_degraded}",
                        className="text-muted",
                    ),
                ])
            ], style={"borderLeft": f"4px solid {colour}"}),
        ]
    else:
        detail = [html.P("Click Assess to run analysis.",
                         className="text-muted")]

    # History chart
    fig_history = go.Figure()
    for e in MONITORED_ENTITIES:
        history = risk_history.get(e, [])
        if history:
            fig_history.add_trace(go.Scatter(
                x=[h["timestamp"] for h in history],
                y=[h["score"] for h in history],
                name=e,
                line=dict(color=ENTITY_COLOURS.get(e, "#888"), width=2),
                mode="lines+markers",
            ))

    fig_history.add_hrect(y0=80, y1=100, fillcolor="rgba(255,68,68,0.1)",
                          line_width=0, annotation_text="ESCALATE")
    fig_history.add_hrect(y0=60, y1=80, fillcolor="rgba(255,140,0,0.1)",
                          line_width=0, annotation_text="ALERT")
    fig_history.add_hrect(y0=40, y1=60, fillcolor="rgba(255,215,0,0.1)",
                          line_width=0, annotation_text="WATCH")
    fig_history.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105], title="ARIA Score"),
        height=300,
        margin=dict(l=40, r=20, t=20, b=30),
        legend=dict(orientation="h", y=-0.3),
    )

    # Summary table
    rows = []
    for e in MONITORED_ENTITIES:
        r = risk_cache.get(e)
        if r:
            colour = RISK_COLOURS.get(r.risk_level.value, "#888")
            rows.append(html.Tr([
                html.Td(e),
                html.Td(f"{r.aria_risk_score:.1f}",
                        style={"color": colour, "fontWeight": "bold"}),
                html.Td(risk_badge(r.risk_level.value)),
                html.Td("⚠️ Yes" if r.requires_human_review else "—",
                        style={"color": "#FF4444"
                               if r.requires_human_review else "#888"}),
                html.Td(f"{r.processing_time_ms:.0f}ms",
                        className="text-muted"),
            ], style={"cursor": "pointer"}))

    table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Entity"), html.Th("Score"),
            html.Th("Level"), html.Th("Human Review"),
            html.Th("Latency"),
        ]))] + [html.Tbody(rows)],
        bordered=True, hover=True, striped=True,
        style={"fontSize": "0.85rem"},
    )

    last_updated = f"Last updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}"

    return [selected] + gauges + [detail, fig_history, table, last_updated]


if __name__ == "__main__":
    app.run(debug=False, port=8056)