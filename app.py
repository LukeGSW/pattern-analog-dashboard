"""
app.py — Analoghi Storici di Configurazione OHLC
================================================
Dato un asset EODHD, mappa le ultime N barre daily, ritrova nello storico le
configurazioni piu' simili (forma OHLC normalizzata, scale-invariant) e mostra cosa
e' successo dopo a piu' orizzonti, confrontandolo con la baseline incondizionata.

La logica vive in src/; qui c'e' solo orchestrazione e UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    COLORS,
    candlestick_query,
    forward_fan,
    normalized_overlay,
    returns_histogram,
)
from src.data_fetcher import fetch_ohlc
from src.outcomes import (
    baseline_path_quantiles,
    build_signal_table,
    compute_robustness,
    forward_paths,
    horizon_analysis,
    matches_table,
)
from src.patterns import find_analogs, make_bar_weights, normalize_window

# ---------------------------------------------------------------------------
# Configurazione pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analoghi Storici OHLC | Kriterion Quant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_api_key():
    """Legge la chiave EODHD dai Secrets (Streamlit Cloud o .streamlit/secrets.toml)."""
    try:
        return st.secrets["EODHD_API_KEY"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar — parametri
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parametri")
    ticker = st.text_input(
        "Ticker EODHD", value="SPY.US",
        help="Esempi: SPY.US (azioni/ETF) · GSPC.INDX (indici) · "
             "EURUSD.FOREX (forex) · BTC-USD.CC (crypto)",
    )
    asset_class = st.selectbox("Classe asset", ["Azioni", "Indici", "Forex", "Crypto"])
    start_dt = st.date_input("Storico da", value=pd.Timestamp("2008-01-01"))

    st.divider()
    st.subheader("Configurazione del match")
    pattern_len = st.slider("Lunghezza pattern (barre)", 3, 10, 5)
    percentile = st.slider("Soglia adattiva — percentile più simile (%)",
                           0.5, 10.0, 2.0, 0.5,
                           help="Seleziona le finestre più vicine nello storico di "
                                "QUESTO asset. Più bassa = match più fedeli ma meno numerosi.")
    min_matches = st.slider("Match minimi", 10, 100, 30, 5)
    max_matches = st.slider("Match massimi", 50, 500, 200, 50)
    quality_floor = st.slider("Pavimento di qualità (similarità %)", 50, 95, 75, 5,
                              help="Sotto questa soglia gli analoghi sono considerati deboli.")
    recency_decay = st.slider(
        "Decadimento recenza (1 = pesi uguali)", 0.60, 1.00, 1.00, 0.05,
        help="Sotto 1 le barre più recenti pesano di più nel match: rende il risultato "
             "più stabile quando si allunga la finestra.",
    )

    st.divider()
    st.caption("Dati: EODHD (adjusted) · cache 1h")

# Orizzonti per classe: crypto gira 7gg/settimana, le altre seguono il calendario di borsa.
horizons = [7, 30, 90] if asset_class == "Crypto" else [5, 20, 63]
max_horizon = max(horizons)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧭 Analoghi Storici di Configurazione OHLC")
st.markdown(
    """
Questa dashboard prende la **fotografia delle ultime barre** di un asset, ne cerca nello
storico le configurazioni **più simili per forma** (OHLC normalizzato, indipendente dal
livello di prezzo) e mostra **cosa è successo dopo** a più orizzonti. Confrontando la
distribuzione *condizionata* (dopo configurazioni simili) con quella *incondizionata*
(qualsiasi giorno), si capisce se quel pattern porta con sé un comportamento ricorrente.

> **Come si usa:** inserisci il ticker e la classe nella sidebar, regola la soglia di
> similarità, e leggi le sezioni A → B → C.
"""
)
st.divider()

# ---------------------------------------------------------------------------
# API key + download dati
# ---------------------------------------------------------------------------
api_key = get_api_key()
if not api_key:
    st.error(
        "Chiave **EODHD_API_KEY** non trovata nei Secrets. "
        "Configurala in Streamlit Cloud (Settings → Secrets) o in "
        "`.streamlit/secrets.toml` in locale. Vedi il README."
    )
    st.stop()

today = pd.Timestamp.today().strftime("%Y-%m-%d")
with st.spinner(f"Scarico {ticker} da EODHD…"):
    try:
        df = fetch_ohlc(ticker.strip(), str(start_dt), today, api_key)
    except Exception as e:  # noqa: BLE001 — vogliamo un messaggio leggibile in UI
        st.error(f"Errore nel download dati: {e}. Verifica ticker e chiave API.")
        st.stop()

min_bars_needed = pattern_len + max_horizon + 50
if df is None or df.empty or len(df) < min_bars_needed:
    n = 0 if df is None else len(df)
    st.error(
        f"Dati insufficienti per **{ticker}**: {n} barre disponibili, ne servono almeno "
        f"~{min_bars_needed}. Prova un ticker con storia più lunga o una data di inizio precedente."
    )
    st.stop()

st.caption(
    f"**{ticker}** · {len(df):,} barre · dal {df.index[0].date()} al {df.index[-1].date()} · "
    f"orizzonti: {', '.join(str(h) + 'b' for h in horizons)}"
)

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
ohlc = df[["open", "high", "low", "close"]].values
N = len(df)
last_valid_end = N - 1 - max_horizon

try:
    mr = find_analogs(
        ohlc,
        pattern_len=pattern_len,
        max_horizon=max_horizon,
        percentile=percentile,
        min_matches=min_matches,
        max_matches=max_matches,
        bar_weights=make_bar_weights(pattern_len, recency_decay),
    )
except ValueError as e:
    st.warning(f"Impossibile analizzare la configurazione attuale: {e}")
    st.stop()

if mr.end_indices.size == 0:
    st.warning(
        "Nessun analogo storico utilizzabile. Storia troppo corta per l'orizzonte scelto "
        "oppure configurazione attuale anomala (range nullo)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI + avvisi di affidabilità
# ---------------------------------------------------------------------------
last_price = float(df["close"].iloc[-1])
ret_1d = (last_price / float(df["close"].iloc[-2]) - 1.0) * 100.0
best_sim = float(mr.similarities.max())
med_sim = float(np.median(mr.similarities))
n_found = int(mr.end_indices.size)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultimo prezzo", f"{last_price:,.2f}", f"{ret_1d:+.2f}%")
c2.metric("Analoghi trovati", f"{n_found}", help=f"su {mr.n_candidates:,} finestre valutate")
c3.metric("Miglior somiglianza", f"{best_sim:.1f}%")
c4.metric("Somiglianza mediana", f"{med_sim:.1f}%")

if n_found < min_matches:
    st.warning(
        f"Solo **{n_found}** match indipendenti (< {min_matches}): campione esiguo, "
        "interpreta le statistiche con cautela."
    )
if best_sim < quality_floor:
    st.warning(
        f"Anche il match migliore è sotto il pavimento di qualità "
        f"(**{best_sim:.1f}% < {quality_floor}%**): gli analoghi sono deboli, le statistiche "
        "forward potrebbero non essere significative."
    )
elif med_sim < quality_floor:
    st.info(
        f"Somiglianza mediana sotto il pavimento ({med_sim:.1f}% < {quality_floor}%): "
        "qualità dei match disomogenea."
    )

st.divider()

# ---------------------------------------------------------------------------
# A — La configurazione attuale
# ---------------------------------------------------------------------------
st.subheader("A — La configurazione attuale")
st.markdown(
    f"Le ultime **{pattern_len} barre** di {ticker}: è questa la 'fotografia' che cerchiamo "
    "di ritrovare nello storico. Il matching avviene sulla **forma** OHLC normalizzata, "
    "non sui prezzi assoluti."
)
last_bars = df.iloc[-pattern_len:]
st.plotly_chart(
    candlestick_query(
        last_bars.index, last_bars["open"], last_bars["high"],
        last_bars["low"], last_bars["close"],
        f"{ticker} — ultime {pattern_len} barre",
    ),
    width="stretch",
)

# ---------------------------------------------------------------------------
# B — Gli analoghi storici
# ---------------------------------------------------------------------------
st.subheader("B — Gli analoghi storici")
st.markdown(
    "A sinistra, la traiettoria *close* normalizzata della configurazione attuale (arancio) "
    "sovrapposta al fascio degli analoghi trovati: più il fascio è stretto attorno alla linea "
    "arancio, più i match sono fedeli. Sotto, l'elenco dei singoli analoghi con la loro data, "
    "la somiglianza e i rendimenti che hanno prodotto."
)

match_vecs = [normalize_window(ohlc[e - pattern_len + 1:e + 1]) for e in mr.end_indices]
match_vecs = [v for v in match_vecs if v is not None]
st.plotly_chart(
    normalized_overlay(mr.query_vector, match_vecs, pattern_len,
                       "Forma normalizzata: configurazione attuale vs analoghi"),
    width="stretch",
)

mt = matches_table(df, mr, horizons)
ret_cols = [c for c in mt.columns if c.startswith("+")]
styled = (
    mt.style
    .format({"Similarità %": "{:.1f}", **{c: "{:+.2f}" for c in ret_cols}})
    .background_gradient(cmap="RdYlGn", subset=ret_cols, vmin=-15, vmax=15)
    .background_gradient(cmap="Blues", subset=["Similarità %"])
)
st.dataframe(styled, width="stretch", height=300)

st.divider()

# ---------------------------------------------------------------------------
# C — Cosa è successo dopo
# ---------------------------------------------------------------------------
st.subheader("C — Cosa è successo dopo")
st.markdown(
    "Per ogni orizzonte confrontiamo la distribuzione dei rendimenti **dopo le configurazioni "
    "simili** (condizionata) con quella di **un giorno qualsiasi** (baseline incondizionata). "
    "Se le due distribuzioni coincidono, il pattern non aggiunge informazione; se differiscono "
    "in media, mediana o % di casi positivi, c'è un comportamento ricorrente."
)

summary_df, details = horizon_analysis(df, mr.end_indices, horizons, last_valid_end)
signal_df, signals = build_signal_table(details, horizons)
sig_by_h = {s["h"]: s for s in signals}

# Sintesi immediata: gli orizzonti con edge concorde (verde/rosso) in testata.
strong = [s for s in signals if s["emoji"] in ("🟢", "🔴")]
if strong:
    txt = " · ".join(f"**{s['h']} barre** {s['emoji']} {s['label']}" for s in strong)
    st.success(f"Edge concorde rispetto alla base → {txt}")
else:
    st.info(
        "Nessun orizzonte mostra un edge concorde: dopo questa configurazione il "
        "comportamento è in linea con la base o ambiguo (vedi colonna **Segnale**)."
    )

# Tabella 'edge-first': valori condizionati + delta vs base + verdetto.
st.dataframe(
    signal_df.style
    .format({"Mediana %": "{:+.2f}", "Media %": "{:+.2f}", "% positivi": "{:.1f}",
             "MFE %": "{:+.2f}", "MAE %": "{:+.2f}",
             "Δ mediana": "{:+.2f}", "Δ media": "{:+.2f}", "Δ % pos.": "{:+.1f}"})
    .background_gradient(cmap="RdYlGn", subset=["Δ mediana", "Δ media"], vmin=-3, vmax=3)
    .background_gradient(cmap="RdYlGn", subset=["Δ % pos."], vmin=-10, vmax=10),
    width="stretch",
)
st.caption(
    "🟢 le tre misure concordano · 🟡 segnali in conflitto (spesso la media inganna per "
    "asimmetria) · 🔴 concorde al ribasso · ⚪ in linea con la base. "
    "I **Δ** sono la differenza rispetto alla baseline: è lì che sta l'edge."
)

with st.expander("📋 Dettaglio completo: condizionato vs baseline"):
    num_cols = [c for c in summary_df.columns if c not in ("Orizzonte", "N match")]
    st.dataframe(
        summary_df.style.format({c: "{:+.2f}" for c in num_cols}),
        width="stretch",
    )

close_arr = df["close"].values
tabs = st.tabs([f"{h} barre" for h in horizons])
for tab, h in zip(tabs, horizons):
    with tab:
        d = details[h]
        sg = sig_by_h[h]
        st.markdown(f"### Segnale a {h} barre: {sg['emoji']} {sg['label']}")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                returns_histogram(d["cond"], d["base"], f"{h} barre",
                                  f"Distribuzione rendimenti a {h} barre"),
                width="stretch",
            )
        with col2:
            paths = forward_paths(close_arr, mr.end_indices, h)
            bq = baseline_path_quantiles(close_arr, h, last_valid_end)
            st.plotly_chart(
                forward_fan(paths, bq, h, f"Traiettorie forward a {h} barre"),
                width="stretch",
            )

        if d["cond"].size:
            cmean, bmean = float(np.mean(d["cond"])), float(np.mean(d["base"]))
            cpos = float(np.mean(d["cond"] > 0) * 100)
            bpos = float(np.mean(d["base"] > 0) * 100)
            edge = cmean - bmean
            verdict = "📈 sopra" if edge > 0 else "📉 sotto"
            st.markdown(
                f"A **{h} barre** — media condizionata **{cmean:+.2f}%** vs baseline "
                f"{bmean:+.2f}% ({verdict} la baseline di {edge:+.2f} pp) · "
                f"casi positivi **{cpos:.0f}%** vs {bpos:.0f}% · "
                f"MFE medio {float(np.mean(d['mfe'])):+.2f}% / MAE medio {float(np.mean(d['mae'])):+.2f}%."
            )

st.divider()

# ---------------------------------------------------------------------------
# Robustezza del segnale al variare di L
# ---------------------------------------------------------------------------
st.subheader("D — Robustezza al variare della finestra")
st.markdown(
    "Se il verdetto cambia molto aggiungendo o togliendo candele, l'edge **non è affidabile**. "
    "Qui ricalcoliamo il segnale per una banda di lunghezze **L = 4…12** e ne misuriamo la "
    "stabilità: una riga tutta dello stesso colore = robusto; una riga 'arcobaleno' = instabile. "
    "Fidati solo dei segnali coerenti lungo la banda."
)
run_robust = st.checkbox("Esegui analisi di robustezza (più lenta: ricalcola su 9 finestre)")
if run_robust:
    l_values = list(range(4, 13))
    with st.spinner("Calcolo il segnale su L = 4…12…"):
        grid_df, quality = compute_robustness(
            ohlc, df, l_values, horizons, last_valid_end, max_horizon,
            percentile, min_matches, max_matches, recency_decay,
        )
    st.dataframe(grid_df, width="stretch")
    st.caption(
        "🟢 rialzista · 🔴 ribassista · 🟡 ambiguo/discorde · ⚪ in linea con la base · "
        "⚠️ direzione instabile. La colonna **Robustezza** sintetizza la stabilità della riga."
    )

    quality_df = pd.DataFrame({
        "L": list(quality.keys()),
        "N match": [quality[k][0] for k in quality],
        "Somiglianza mediana %": [quality[k][1] for k in quality],
    })
    st.markdown(
        "**Numerosità e qualità dei match al crescere di L.** Con la soglia adattiva il "
        "numero di match resta circa costante, mentre la **somiglianza tende a calare**: "
        "allungare la finestra scambia contesto in cambio di fedeltà."
    )
    st.dataframe(
        quality_df.style.format({"Somiglianza mediana %": "{:.1f}"})
        .background_gradient(cmap="Blues", subset=["Somiglianza mediana %"]),
        width="stretch", hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Metodologia
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Metodologia e limiti"):
    st.markdown(
        f"""
- **Dati**: OHLC daily *adjusted* da EODHD (open/high/low riscalati col fattore
  `adjusted_close / close` per coerenza su split e dividendi).
- **Configurazione**: ultime **{pattern_len} barre** → vettore OHLC normalizzato
  *scale-invariant*: ancoraggio all'open della prima barra, scala = range della finestra
  (max High − min Low). Scelta **anomaly-aware**: una barra estrema resta un tratto distintivo
  del pattern invece di essere appiattita (come accadrebbe normalizzando per ATR).
- **Similarità**: distanza euclidea nello spazio normalizzato, riportata come
  `100 × (1 − rmse)`, dove l'rmse è lo scarto medio per-punto in unità di range (la finestra
  ha estensione verticale ≈ 1, quindi la % è direttamente interpretabile).
- **Selezione**: soglia **adattiva a percentile** (top {percentile:.1f}% più simili *di questo
  asset*), con **diradamento** delle finestre sovrapposte (stesso evento contato una volta),
  numerosità minima {min_matches} e cap {max_matches}, e **pavimento di qualità** {quality_floor}%.
- **Esiti**: rendimenti reali (%) a {', '.join(str(h) for h in horizons)} barre, più MFE/MAE
  (massima escursione favorevole/avversa). Le finestre candidate escludono gli ultimi
  {max_horizon} giorni (orizzonte forward non ancora completo → niente look-ahead).
- **Baseline**: distribuzione *incondizionata* degli stessi rendimenti forward su ogni barra
  dell'asset, per misurare se il pattern fa davvero la differenza.

**Limiti da tenere a mente**
- Studio **descrittivo**, non un test di significatività: differenze piccole possono essere caso.
- Match temporalmente vicini conservano autocorrelazione residua anche dopo il diradamento.
- Regimi di mercato diversi (volatilità, tassi) non sono separati: un analogo del 2008 e uno
  del 2017 vivono in contesti differenti.
- Su asset con storia breve (molte crypto) il campione può essere troppo piccolo: leggi sempre
  numerosità e somiglianza prima delle conclusioni.
"""
    )

st.caption("Kriterion Quant · Analoghi storici OHLC · dati EODHD")
