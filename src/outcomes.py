"""
outcomes.py
-----------
Esiti forward dei match e baseline incondizionata.

Per ogni match (chiusura della finestra all'indice e) si misura cosa e' successo
dopo a 5/20/63 barre (o 7/30/90 per crypto). La baseline e' la distribuzione
INCONDIZIONATA degli stessi rendimenti forward calcolata su ogni barra dell'asset:
serve a capire se il comportamento dopo la configurazione e' davvero diverso dal caso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: np.ndarray, end_indices: np.ndarray, horizon: int) -> np.ndarray:
    """Rendimenti % forward a 'horizon' barre dalla chiusura di ogni end-index."""
    e = np.asarray(end_indices)
    e = e[e + horizon < close.size]
    if e.size == 0:
        return np.array([])
    return (close[e + horizon] / close[e] - 1.0) * 100.0


def excursions(high, low, close, end_indices, horizon):
    """
    MFE e MAE % nell'intervallo (e, e+horizon] rispetto al close di ingresso.

    Ritorna (mfe, mae): massima escursione favorevole e avversa per ciascun match.
    """
    mfe, mae = [], []
    n = close.size
    for e in end_indices:
        if e + horizon < n:
            entry = close[e]
            hh = high[e + 1:e + horizon + 1].max()
            ll = low[e + 1:e + horizon + 1].min()
            mfe.append((hh / entry - 1.0) * 100.0)
            mae.append((ll / entry - 1.0) * 100.0)
    return np.array(mfe), np.array(mae)


def baseline_returns(close: np.ndarray, horizon: int, last_valid_end: int) -> np.ndarray:
    """Distribuzione incondizionata dei rendimenti forward a 'horizon' barre."""
    e = np.arange(0, last_valid_end + 1)
    e = e[e + horizon < close.size]
    if e.size == 0:
        return np.array([])
    return (close[e + horizon] / close[e] - 1.0) * 100.0


def forward_paths(close: np.ndarray, end_indices: np.ndarray, horizon: int) -> np.ndarray:
    """Traiettorie cumulate % da t=0 a t=horizon per ogni match. shape (n_match, horizon+1)."""
    paths = []
    n = close.size
    for e in end_indices:
        if e + horizon < n:
            seg = close[e:e + horizon + 1]
            paths.append((seg / seg[0] - 1.0) * 100.0)
    return np.array(paths) if paths else np.empty((0, horizon + 1))


def baseline_path_quantiles(close, horizon, last_valid_end, qs=(25, 50, 75)):
    """Quantili della traiettoria forward incondizionata (overlay di riferimento)."""
    e = np.arange(0, last_valid_end + 1)
    e = e[e + horizon < close.size]
    if e.size == 0:
        return None
    M = np.vstack([(close[i:i + horizon + 1] / close[i] - 1.0) * 100.0 for i in e])
    return {q: np.percentile(M, q, axis=0) for q in qs}


def summarize(returns: np.ndarray) -> dict:
    """Statistiche descrittive di una distribuzione di rendimenti %."""
    if returns.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, pct_pos=np.nan,
                    std=np.nan, p25=np.nan, p75=np.nan)
    return dict(
        n=int(returns.size),
        mean=float(np.mean(returns)),
        median=float(np.median(returns)),
        pct_pos=float(np.mean(returns > 0) * 100.0),
        std=float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        p25=float(np.percentile(returns, 25)),
        p75=float(np.percentile(returns, 75)),
    )


def horizon_analysis(df_ohlc: pd.DataFrame, end_indices, horizons, last_valid_end):
    """
    Tabella riassuntiva (condizionato vs baseline) per ogni orizzonte + dettagli grezzi.

    Ritorna
    -------
    (summary_df, details)
        summary_df : DataFrame con una riga per orizzonte.
        details    : dict {horizon: {'cond', 'base', 'mfe', 'mae'}} con gli array grezzi
                     per i grafici.
    """
    close = df_ohlc["close"].values
    high = df_ohlc["high"].values
    low = df_ohlc["low"].values

    rows, details = [], {}
    for h in horizons:
        cond = forward_returns(close, end_indices, h)
        base = baseline_returns(close, h, last_valid_end)
        mfe, mae = excursions(high, low, close, end_indices, h)
        cs, bs = summarize(cond), summarize(base)
        rows.append({
            "Orizzonte": f"{h} barre",
            "N match": cs["n"],
            "Media cond. %": cs["mean"],
            "Media base %": bs["mean"],
            "Mediana cond. %": cs["median"],
            "Mediana base %": bs["median"],
            "% positivi cond.": cs["pct_pos"],
            "% positivi base": bs["pct_pos"],
            "Dev.std cond. %": cs["std"],
            "MFE med. %": float(np.mean(mfe)) if mfe.size else np.nan,
            "MAE med. %": float(np.mean(mae)) if mae.size else np.nan,
        })
        details[h] = dict(cond=cond, base=base, mfe=mfe, mae=mae)
    return pd.DataFrame(rows), details


def matches_table(df_ohlc: pd.DataFrame, mr, horizons) -> pd.DataFrame:
    """Tabella dei singoli match: data, similarita' e rendimenti forward per orizzonte."""
    close = df_ohlc["close"].values
    dates = df_ohlc.index
    rows = []
    for e, sim in zip(mr.end_indices, mr.similarities):
        row = {"Data analogo": dates[e].date(), "Similarità %": sim}
        for h in horizons:
            row[f"+{h}b %"] = (close[e + h] / close[e] - 1.0) * 100.0 if e + h < close.size else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Similarità %", ascending=False).reset_index(drop=True)
    return out
