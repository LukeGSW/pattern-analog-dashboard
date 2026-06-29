"""
data_fetcher.py
---------------
Download e caching dei dati OHLC daily da EODHD.

I prezzi vengono restituiti ADJUSTED su tutta la candela: EODHD aggiusta solo
'adjusted_close', quindi qui ricostruiamo open/high/low coerenti applicando il
fattore di aggiustamento (adjusted_close / close). Cosi' split e dividendi non
generano falsi pattern sui gap tecnici.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

EOD_URL = "https://eodhd.com/api/eod/{ticker}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc(ticker: str, from_date: str, to_date: str, api_key: str) -> pd.DataFrame:
    """
    Scarica OHLC daily adjusted da EODHD.

    Parametri
    ---------
    ticker : str
        Simbolo in formato EODHD. Es: 'SPY.US', 'GSPC.INDX', 'EURUSD.FOREX', 'BTC-USD.CC'.
    from_date, to_date : str
        Date in formato 'YYYY-MM-DD'.
    api_key : str
        Chiave EODHD (entra nella chiave di cache, cosi' chiavi diverse non collidono).

    Ritorno
    -------
    pd.DataFrame
        Indicizzato per data (crescente), colonne [open, high, low, close, volume],
        tutte adjusted. DataFrame vuoto se l'API non restituisce dati.
    """
    url = EOD_URL.format(ticker=ticker)
    params = {
        "from": from_date,
        "to": to_date,
        "period": "d",
        "api_token": api_key,
        "fmt": "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Fattore di aggiustamento: riscala open/high/low come adjusted_close riscala close.
    if "adjusted_close" in df.columns:
        factor = df["adjusted_close"] / df["close"]
        close_adj = df["adjusted_close"]
    else:
        factor = 1.0
        close_adj = df["close"]

    out = pd.DataFrame(index=df.index)
    out["open"] = df["open"] * factor
    out["high"] = df["high"] * factor
    out["low"] = df["low"] * factor
    out["close"] = close_adj
    out["volume"] = df["volume"] if "volume" in df.columns else 0.0

    return out.dropna(subset=["open", "high", "low", "close"])
