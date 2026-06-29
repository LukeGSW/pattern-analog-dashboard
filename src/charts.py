"""
charts.py
---------
Funzioni grafiche Plotly riutilizzabili (tema scuro coerente) per la dashboard.
Ricevono dati gia' calcolati: non fanno fetch ne' calcoli pesanti.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

COLORS = {
    "primary":    "#2196F3",
    "secondary":  "#FF9800",
    "positive":   "#4CAF50",
    "negative":   "#F44336",
    "neutral":    "#9E9E9E",
    "background": "#1E1E2E",
    "surface":    "#2A2A3E",
    "text":       "#E0E0E0",
    "accent":     "#AB47BC",
}


def get_layout(title: str, xaxis_title: str = "", yaxis_title: str = "") -> dict:
    """Layout professionale riutilizzabile per tutti i grafici Plotly (dark theme)."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif"),
        xaxis=dict(title=xaxis_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="#333355",
                   zeroline=False, color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444466"),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=60),
    )


def candlestick_query(dates, o, h, low, c, title: str) -> go.Figure:
    """Candlestick delle ultime L barre (la configurazione attuale)."""
    fig = go.Figure(go.Candlestick(
        x=list(dates), open=o, high=h, low=low, close=c,
        increasing_line_color=COLORS["positive"],
        decreasing_line_color=COLORS["negative"],
        name="OHLC",
    ))
    fig.update_layout(**get_layout(title, "Data", "Prezzo (adjusted)"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def normalized_overlay(query_vec, match_vecs, pattern_len: int, title: str) -> go.Figure:
    """
    Overlay delle traiettorie close normalizzate: configurazione attuale (in evidenza)
    sopra il fascio degli analoghi storici. Mostra quanto i match "assomigliano" alla
    forma attuale nello spazio scale-invariant.
    """
    x = list(range(1, pattern_len + 1))
    fig = go.Figure()

    for mv in match_vecs:
        close_path = mv.reshape(pattern_len, 4)[:, 3]
        fig.add_trace(go.Scatter(
            x=x, y=close_path, mode="lines",
            line=dict(color="rgba(120,140,210,0.18)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    q_close = query_vec.reshape(pattern_len, 4)[:, 3]
    fig.add_trace(go.Scatter(
        x=x, y=q_close, mode="lines+markers",
        line=dict(color=COLORS["secondary"], width=3),
        marker=dict(size=7), name="Configurazione attuale",
    ))

    fig.update_layout(**get_layout(title, "Barra", "Close normalizzato (range-unit)"))
    fig.update_layout(hovermode="closest")
    return fig


def returns_histogram(cond: np.ndarray, base: np.ndarray, horizon_label: str, title: str) -> go.Figure:
    """Distribuzione dei rendimenti forward: condizionato (match) vs baseline."""
    fig = go.Figure()
    if base.size:
        fig.add_trace(go.Histogram(
            x=base, name="Baseline (incond.)", histnorm="probability density",
            marker_color=COLORS["neutral"], opacity=0.45, nbinsx=45,
        ))
    if cond.size:
        fig.add_trace(go.Histogram(
            x=cond, name="Condizionato (match)", histnorm="probability density",
            marker_color=COLORS["primary"], opacity=0.65, nbinsx=45,
        ))
        fig.add_vline(
            x=float(np.median(cond)), line=dict(color=COLORS["secondary"], dash="dash"),
            annotation_text="mediana cond.", annotation_position="top",
        )
    fig.update_layout(**get_layout(title, f"Rendimento a {horizon_label} (%)", "Densità"))
    fig.update_layout(barmode="overlay", hovermode="x")
    return fig


def forward_fan(paths: np.ndarray, base_quantiles, horizon: int, title: str) -> go.Figure:
    """
    Fascio delle traiettorie forward dei match (allineate a t=0) con mediana e banda
    25-75%, piu' la mediana della baseline incondizionata come riferimento.
    """
    x = list(range(0, horizon + 1))
    fig = go.Figure()

    for p in paths:
        fig.add_trace(go.Scatter(
            x=x, y=p, mode="lines",
            line=dict(color="rgba(33,150,243,0.10)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    if paths.shape[0] > 0:
        med = np.median(paths, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", fill="tonexty",
                                 fillcolor="rgba(33,150,243,0.20)", line=dict(width=0),
                                 name="25–75% match"))
        fig.add_trace(go.Scatter(x=x, y=med, mode="lines",
                                 line=dict(color=COLORS["primary"], width=3),
                                 name="Mediana match"))

    if base_quantiles is not None:
        fig.add_trace(go.Scatter(
            x=x, y=base_quantiles[50], mode="lines",
            line=dict(color=COLORS["neutral"], width=2, dash="dash"),
            name="Mediana baseline",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["neutral"], width=1))
    fig.update_layout(**get_layout(title, "Barre dopo la configurazione", "Rendimento cumulato (%)"))
    fig.update_layout(hovermode="x unified")
    return fig
