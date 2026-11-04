"""Dashboard charts (Plotly). Colours follow a validated categorical order + fixed status colours."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"Resolved": "#0ca30c", "Escalated": "#d03b3b", "Filtered": "#9a9a95"}
URGENCY = {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#104281"}  # ordinal, one hue light -> dark
SENTIMENT = {"Positive": "#2a78d6", "Neutral": "#9a9a95", "Negative": "#ec8f8f", "Angry": "#d03b3b"}  # diverging
CATEGORY_ORDER = ["Shipping", "Refund", "Account", "Product", "Billing", "General"]


def _layout(fig: go.Figure, title: str, height: int = 300) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=8, r=8, t=48, b=8), bargap=0.35, bargroupgap=0.08,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        hoverlabel=dict(font_size=12), font=dict(size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False, rangemode="tozero")
    _integer_ticks(fig)
    return fig


def _integer_ticks(fig: go.Figure) -> None:
    """Counts are whole numbers - never show 0.2 tickets."""
    peak = 0
    for tr in fig.data:
        vals = tr.x if getattr(tr, "orientation", None) == "h" else tr.y
        try:
            peak = max(peak, max(float(v) for v in vals))
        except (TypeError, ValueError):
            continue
    if peak <= 10:
        axis = fig.update_xaxes if any(getattr(t, "orientation", None) == "h" for t in fig.data) else fig.update_yaxes
        axis(dtick=1, tick0=0)


def by_category_status(df: pd.DataFrame) -> go.Figure:
    cats = [c for c in CATEGORY_ORDER if c in set(df["category"])] + sorted(set(df["category"]) - set(CATEGORY_ORDER))
    fig = go.Figure()
    for status in ["Resolved", "Escalated"]:
        counts = df[df["status"] == status]["category"].value_counts()
        fig.add_bar(name=status, x=cats, y=[int(counts.get(c, 0)) for c in cats], marker_color=STATUS[status],
                    marker=dict(cornerradius=4, line=dict(width=2, color="rgba(0,0,0,0)")),
                    hovertemplate="%{x}<br>" + status + ": %{y}<extra></extra>")
    fig.update_layout(barmode="stack")
    return _layout(fig, "Tickets per category · resolved vs escalated")


def counts_bar(df: pd.DataFrame, column: str, title: str, colors: dict | None = None, order: list | None = None,
               horizontal: bool = False) -> go.Figure:
    counts = df[column].replace("", "unknown").value_counts()
    keys = [k for k in (order or []) if k in counts.index] + [k for k in counts.index if k not in (order or [])]
    vals = [int(counts[k]) for k in keys]
    cols = [(colors or {}).get(k, SERIES[i % len(SERIES)] if not colors else "#9a9a95") for i, k in enumerate(keys)]
    if horizontal:
        fig = go.Figure(go.Bar(y=keys, x=vals, orientation="h", marker=dict(color=cols, cornerradius=4),
                               text=vals, textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure(go.Bar(x=keys, y=vals, marker=dict(color=cols, cornerradius=4), text=vals,
                               textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
    fig = _layout(fig, title)
    fig.update_layout(showlegend=False)
    return fig


def over_time(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["ts"])
    span_hours = (d["ts"].max() - d["ts"].min()).total_seconds() / 3600 if len(d) > 1 else 0
    freq, label = ("D", "day") if span_hours > 48 else ("h", "hour")
    d["bucket"] = d["ts"].dt.floor(freq)
    fig = go.Figure()
    for status in ["Resolved", "Escalated"]:
        s = d[d["status"] == status].groupby("bucket").size()
        fig.add_scatter(name=status, x=s.index, y=s.values, mode="lines+markers",
                        line=dict(width=2, color=STATUS[status]), marker=dict(size=8),
                        hovertemplate="%{x}<br>" + status + ": %{y}<extra></extra>")
    fig = _layout(fig, f"Tickets per {label}")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%d %b %H:%M" if label == "hour" else "%d %b")
    return fig


def escalation_reasons(df: pd.DataFrame) -> go.Figure:
    reasons = Counter()
    for r in df.loc[df["status"] == "Escalated", "escalation_reason"]:
        for part in str(r).split(";"):
            part = part.strip()
            if part.startswith("Low resolution confidence"):
                part = "Low resolution confidence"
            if part.startswith("Log Agent:"):
                part = "Log Agent judgement"
            if part:
                reasons[part] += 1
    keys = [k for k, _ in reasons.most_common(8)]
    fig = go.Figure(go.Bar(y=keys, x=[reasons[k] for k in keys], orientation="h",
                           marker=dict(color=STATUS["Escalated"], cornerradius=4), text=[reasons[k] for k in keys],
                           textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "Why tickets were escalated", height=max(240, 46 * len(keys) + 80))
    fig.update_layout(showlegend=False)
    return fig


def by_product_line_status(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["line"] = d["product_line_name"].replace("", "General (orders, account, payments)")
    lines = d["line"].value_counts().index.tolist()
    fig = go.Figure()
    for status in ["Resolved", "Escalated"]:
        counts = d[d["status"] == status]["line"].value_counts()
        fig.add_bar(name=status, y=lines, x=[int(counts.get(l, 0)) for l in lines], orientation="h",
                    marker=dict(color=STATUS[status], cornerradius=4),
                    hovertemplate="%{y}<br>" + status + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Tickets per product line / service", height=max(300, 40 * len(lines) + 90))


def desk_workload(df: pd.DataFrame) -> go.Figure:
    open_ = df[(df["status"] == "Escalated") & (df["review_status"] == "Pending")]
    counts = open_["assigned_team"].replace("", "Support").value_counts()
    keys = counts.index.tolist()
    fig = go.Figure(go.Bar(y=keys, x=[int(counts[k]) for k in keys], orientation="h",
                           marker=dict(color=SERIES[0], cornerradius=4), text=[int(counts[k]) for k in keys],
                           textposition="outside", hovertemplate="%{y}: %{x} open<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "Open escalations per specialist desk", height=max(300, 40 * len(keys) + 90))
    fig.update_layout(showlegend=False)
    return fig


def guardrail_blocks(filtered: pd.DataFrame) -> go.Figure:
    reasons = Counter()
    for g in filtered["guardrail"]:
        for part in str(g).split(";"):
            part = part.strip()
            if part.startswith("blocked:"):
                reasons[part.split(":", 1)[1].replace("_", " ")] += 1
    keys = [k for k, _ in reasons.most_common(8)]
    fig = go.Figure(go.Bar(y=keys, x=[reasons[k] for k in keys], orientation="h",
                           marker=dict(color=STATUS["Filtered"], cornerradius=4), text=[reasons[k] for k in keys],
                           textposition="outside", hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "Messages stopped by the guardrail", height=max(240, 46 * len(keys) + 80))
    fig.update_layout(showlegend=False)
    return fig
