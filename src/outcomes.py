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


# ---------------------------------------------------------------------------
# Indicatore di concordanza: verdetto DEFINITO per ogni orizzonte
# ---------------------------------------------------------------------------
# Soglie di materialità (dead-zone): sotto queste un edge è considerato trascurabile,
# così il rumore non viene scambiato per segnale.
HIT_TOL_PP = 3.0          # hit-rate: punti percentuali
RET_TOL_STD_FRAC = 0.10   # rendimenti: frazione della dev.std condizionata
RET_TOL_FLOOR = 0.10      # rendimenti: pavimento minimo (punti percentuali)


def _sign(edge: float, tol: float) -> int:
    """Segno dell'edge con zona morta: +1 / -1 / 0 (trascurabile o non finito)."""
    if not np.isfinite(edge):
        return 0
    if edge > tol:
        return 1
    if edge < -tol:
        return -1
    return 0


def _verdict(s_med: int, s_hit: int, s_mean: int):
    """
    Mappa ESAUSTIVA dei tre segni in un verdetto definito (emoji, etichetta).

    Mediana e hit-rate sono le metriche robuste (la loro somma è 'robust'); la media è
    secondaria perché può ingannare per asimmetria. Copre tutte le 27 combinazioni.
    """
    robust = s_med + s_hit
    if robust == 2:
        return ("🟢", "Rialzista concorde") if s_mean >= 0 \
            else ("🟢", "Rialzista (media in controtendenza)")
    if robust == -2:
        return ("🔴", "Ribassista concorde") if s_mean <= 0 \
            else ("🔴", "Ribassista (media in controtendenza)")
    if robust == 1:
        if s_mean >= 0:
            return ("🟢", "Rialzista debole")
        leader = "hit-rate" if s_hit > 0 else "mediana"
        return ("🟡", f"Discorde: {leader} ↑ ma media ↓")
    if robust == -1:
        if s_mean <= 0:
            return ("🔴", "Ribassista debole")
        leader = "hit-rate" if s_hit < 0 else "mediana"
        return ("🟡", f"Discorde: {leader} ↓ ma media ↑")
    # robust == 0
    if s_med == 0 and s_hit == 0:
        if s_mean > 0:
            return ("🟡", "Solo la media sopra la base")
        if s_mean < 0:
            return ("🟡", "Solo la media sotto la base")
        return ("⚪", "In linea con la base")
    # mediana e hit-rate di segno opposto → distribuzione asimmetrica
    return ("🟡", "Discorde: mediana e hit-rate divergono")


def classify_signal(cond: np.ndarray, base: np.ndarray) -> dict:
    """
    Verdetto definito per un orizzonte: confronta mediana, hit-rate e media condizionate
    con la baseline e restituisce emoji + etichetta + i delta vs base.
    """
    cs, bs = summarize(cond), summarize(base)
    if cs["n"] == 0 or bs["n"] == 0 or not np.isfinite(bs["median"]):
        return dict(emoji="⚪", label="Dati insufficienti",
                    d_median=np.nan, d_pos=np.nan, d_mean=np.nan)

    d_median = cs["median"] - bs["median"]
    d_pos = cs["pct_pos"] - bs["pct_pos"]
    d_mean = cs["mean"] - bs["mean"]

    ret_tol = max(RET_TOL_STD_FRAC * cs["std"], RET_TOL_FLOOR)
    s_med = _sign(d_median, ret_tol)
    s_hit = _sign(d_pos, HIT_TOL_PP)
    s_mean = _sign(d_mean, ret_tol)

    emoji, label = _verdict(s_med, s_hit, s_mean)
    return dict(emoji=emoji, label=label,
                d_median=d_median, d_pos=d_pos, d_mean=d_mean)


def build_signal_table(details: dict, horizons):
    """
    Tabella riassuntiva 'edge-first' con indicatore di concordanza, costruita dai
    dettagli già calcolati da horizon_analysis (nessuna ricomputazione).

    Ritorna
    -------
    (table_df, signals)
        table_df : DataFrame con valori condizionati, delta vs base e colonna 'Segnale'.
        signals  : lista di dict {h, emoji, label} per la sintesi in testata e i tab.
    """
    rows, signals = [], []
    for h in horizons:
        d = details[h]
        cond, base, mfe, mae = d["cond"], d["base"], d["mfe"], d["mae"]
        cs = summarize(cond)
        sig = classify_signal(cond, base)
        rows.append({
            "Orizzonte": f"{h} barre",
            "N": cs["n"],
            "Mediana %": cs["median"],
            "Δ mediana": sig["d_median"],
            "% positivi": cs["pct_pos"],
            "Δ % pos.": sig["d_pos"],
            "Media %": cs["mean"],
            "Δ media": sig["d_mean"],
            "MFE %": float(np.mean(mfe)) if mfe.size else np.nan,
            "MAE %": float(np.mean(mae)) if mae.size else np.nan,
            "Segnale": f"{sig['emoji']} {sig['label']}",
        })
        signals.append(dict(h=h, emoji=sig["emoji"], label=sig["label"]))
    return pd.DataFrame(rows), signals
