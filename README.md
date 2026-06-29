# 🧭 Analoghi Storici di Configurazione OHLC

Dashboard Streamlit che, dato un asset EODHD, prende la **fotografia delle ultime barre
daily**, ne ritrova nello storico le configurazioni **più simili per forma** (OHLC
normalizzato e *scale-invariant*) e mostra **cosa è successo dopo** a più orizzonti,
confrontando il comportamento *condizionato* con la **baseline incondizionata** dello
stesso asset. Serve a capire se una certa configurazione porta con sé una statistica
ricorrente.

## Come funziona (in breve)

1. **Configurazione** — le ultime *N* barre (default 5) diventano un vettore OHLC
   normalizzato: ancorato all'open della prima barra e diviso per il range della finestra.
   La normalizzazione è *anomaly-aware* (una barra estrema resta significativa, non viene
   appiattita come accadrebbe con l'ATR).
2. **Ricerca analoghi** — distanza euclidea tra la configurazione attuale e ogni finestra
   storica; selezione con **soglia adattiva a percentile** (auto-calibrata sull'asset),
   diradamento delle finestre sovrapposte, numerosità minima e pavimento di qualità.
3. **Esiti forward** — per ogni analogo si misurano i rendimenti reali a 5/20/63 barre
   (7/30/90 per le crypto) più MFE/MAE, e si confrontano con la distribuzione incondizionata.

## Struttura

```
pattern-analog-dashboard/
├── app.py                 # UI Streamlit e orchestrazione
├── requirements.txt
├── .streamlit/
│   ├── config.toml        # tema scuro
│   └── secrets.toml.example
└── src/
    ├── data_fetcher.py    # download EODHD (adjusted) + cache
    ├── patterns.py        # normalizzazione, distanza, selezione match
    ├── outcomes.py        # esiti forward, baseline, statistiche
    └── charts.py          # grafici Plotly
```

## Setup locale

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# inserisci la tua chiave in .streamlit/secrets.toml
streamlit run app.py
```

## Deploy su Streamlit Community Cloud

1. Carica il repository su GitHub.
2. Su [share.streamlit.io](https://share.streamlit.io) crea una nuova app puntando a `app.py`.
3. **Settings → Secrets** e incolla **una sola volta**:
   ```toml
   EODHD_API_KEY = "la_tua_chiave"
   ```
4. Deploy. La chiave resta nei Secrets, non è mai esposta nell'interfaccia né nel codice.

## Formati ticker EODHD

| Classe   | Esempio          |
|----------|------------------|
| Azioni   | `AAPL.US`, `SPY.US` |
| Indici   | `GSPC.INDX` (S&P 500) |
| Forex    | `EURUSD.FOREX`   |
| Crypto   | `BTC-USD.CC`     |

## Parametri (sidebar)

- **Ticker** e **Classe asset** (la classe determina gli orizzonti e il calendario).
- **Storico da** — data di inizio del download.
- **Lunghezza pattern** — quante barre compongono la configurazione (default 5).
- **Soglia adattiva** — percentile delle finestre più simili da considerare match.
- **Match minimi / massimi** — guardrail di numerosità del campione.
- **Pavimento di qualità** — sotto questa similarità gli analoghi sono segnalati come deboli.

## Limiti

Studio **descrittivo**, non un test di significatività. Restano autocorrelazione residua tra
match vicini, mescolanza di regimi di mercato e campioni piccoli su asset con storia breve.
Leggi sempre numerosità e somiglianza prima di trarre conclusioni.

---
*Kriterion Quant · dati EODHD*
