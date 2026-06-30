"""
patterns.py
-----------
Costruzione, normalizzazione e matching delle finestre OHLC.

La "configurazione" delle ultime L barre viene rappresentata come un vettore OHLC
normalizzato (scale-invariant) e confrontata con tutte le finestre storiche dello
stesso asset tramite distanza euclidea. La selezione dei match usa una soglia
ADATTIVA a percentile (auto-calibrata sul singolo asset), con diradamento delle
sovrapposizioni e guardrail di numerosita'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    """Esito del matching: indici, similarita' e metadati diagnostici."""

    end_indices: np.ndarray   # indice posizionale dell'ULTIMA barra di ogni finestra match
    similarities: np.ndarray  # similarita' % in [0, 100]
    distances: np.ndarray     # distanza euclidea nello spazio normalizzato
    n_candidates: int         # numero di finestre storiche valutate
    threshold_distance: float # distanza-soglia derivata dal percentile
    query_vector: np.ndarray  # vettore normalizzato della configurazione attuale


# ---------------------------------------------------------------------------
# Normalizzazione di una singola finestra OHLC
# ---------------------------------------------------------------------------
def normalize_window(ohlc: np.ndarray):
    """
    Rende una finestra OHLC scale-invariant.

    Ancoraggio all'open della prima barra; scala = range complessivo della finestra
    (max High - min Low). La normalizzazione per range (non per ATR) e' "anomaly-aware":
    una barra estrema resta il tratto distintivo del pattern invece di essere appiattita.

    Parametri
    ---------
    ohlc : np.ndarray, shape (L, 4)
        Colonne [open, high, low, close].

    Ritorno
    -------
    np.ndarray shape (4*L,) oppure None se la finestra e' piatta (range nullo).
    """
    o = ohlc[:, 0]
    high = ohlc[:, 1]
    low = ohlc[:, 2]

    anchor = o[0]
    scale = high.max() - low.min()
    if scale <= 0 or not np.isfinite(scale):
        return None

    norm = (ohlc - anchor) / scale          # broadcasting su tutte e 4 le colonne
    return norm.reshape(-1)                  # [o0,h0,l0,c0, o1,h1,l1,c1, ...]


def make_bar_weights(pattern_len: int, decay: float = 1.0) -> np.ndarray:
    """
    Pesi per-barra per la pesatura di recenza nel calcolo della distanza.

    La barra più recente pesa 1, le precedenti `decay**k` (k = distanza dalla più recente).
    I pesi sono normalizzati a somma = pattern_len, così la scala di similarità resta
    invariata. `decay = 1.0` → pesi uniformi (comportamento neutro).

    Una recenza più marcata (decay < 1) rende il match più stabile quando si allunga la
    finestra: le barre aggiunte "in coda" (più vecchie) perturbano meno il risultato.
    """
    L = pattern_len
    if decay >= 1.0:
        return np.ones(L)
    exps = np.arange(L - 1, -1, -1)        # [L-1, ..., 1, 0] → la più recente ha esponente 0
    w = decay ** exps
    return w * (L / w.sum())


def _similarity_from_distance(dist: np.ndarray, n_dims: int) -> np.ndarray:
    """
    Converte la distanza euclidea in similarita' % interpretabile.

    rmse = dist / sqrt(n_dims) e' lo scarto medio per-punto in unita' di range della
    finestra. Per costruzione la finestra ha estensione verticale ~1, quindi
    similarity = 100 * (1 - rmse) si legge come: "in media ogni punto OHLC dista X%
    dell'altezza totale del pattern". Limitata a [0, 100].
    """
    rmse = dist / np.sqrt(n_dims)
    return np.clip(100.0 * (1.0 - rmse), 0.0, 100.0)


def find_analogs(
    ohlc: np.ndarray,
    pattern_len: int = 5,
    max_horizon: int = 63,
    percentile: float = 2.0,
    min_matches: int = 30,
    max_matches: int = 200,
    min_gap: int | None = None,
    bar_weights: np.ndarray | None = None,
) -> MatchResult:
    """
    Trova le finestre storiche piu' simili alla configurazione corrente (ultime L barre).

    Parametri
    ---------
    ohlc : np.ndarray, shape (N, 4)
        OHLC adjusted [open, high, low, close], dal piu' vecchio al piu' recente.
    pattern_len : int
        Lunghezza L della configurazione.
    max_horizon : int
        Massimo orizzonte forward (barre): una finestra candidata deve avere almeno
        questo numero di barre successive per poter misurare gli esiti.
    percentile : float
        Soglia adattiva: seleziona le finestre nel percentile inferiore di distanza
        (es. 2.0 -> il 2% piu' simile). Auto-calibrata sull'asset.
    min_matches, max_matches : int
        Guardrail di numerosita'.
    min_gap : int | None
        Distanza minima (barre) fra gli end-index di due match (diradamento delle
        sovrapposizioni). Default = pattern_len, cioe' finestre non sovrapposte.
    bar_weights : np.ndarray | None
        Pesi per-barra (lunghezza pattern_len) per la pesatura di recenza nella distanza.
        None → pesi uniformi. Vedi make_bar_weights().

    Ritorno
    -------
    MatchResult
    """
    if min_gap is None:
        min_gap = pattern_len
    w = None if bar_weights is None else np.asarray(bar_weights, dtype=float)

    N = ohlc.shape[0]
    L = pattern_len
    empty = MatchResult(np.array([], int), np.array([]), np.array([]), 0, np.nan,
                        np.array([]))

    # Configurazione attuale: ultime L barre.
    query = normalize_window(ohlc[N - L:N])
    if query is None:
        raise ValueError("La configurazione attuale ha range nullo: impossibile normalizzare.")
    n_dims = query.size

    # Finestre candidate: end-index e in [L-1, N-1-max_horizon], cosi' la finestra e'
    # interamente nel passato e ha l'orizzonte forward completo (no look-ahead, no overlap
    # con la configurazione attuale).
    last_valid_end = N - 1 - max_horizon
    if last_valid_end < L - 1:
        return empty

    end_indices = np.arange(L - 1, last_valid_end + 1)
    distances = np.full(end_indices.size, np.nan)
    for k, e in enumerate(end_indices):
        win = normalize_window(ohlc[e - L + 1:e + 1])
        if win is None:
            continue
        if w is None:
            distances[k] = np.sqrt(np.sum((win - query) ** 2))
        else:
            per_bar = ((win - query).reshape(L, 4) ** 2).sum(axis=1)  # scarto² per barra
            distances[k] = np.sqrt(np.sum(w * per_bar))

    valid = np.isfinite(distances)
    end_indices = end_indices[valid]
    distances = distances[valid]
    n_candidates = distances.size
    if n_candidates == 0:
        return empty

    # Soglia adattiva: distanza corrispondente al percentile richiesto.
    threshold_distance = float(np.percentile(distances, percentile))

    # Scorri in ordine di distanza crescente: tieni i match sotto soglia, dirada le
    # sovrapposizioni, fermati a max_matches. Se la soglia rende meno di min_matches,
    # prosegui oltre soglia fino a raggiungere min_matches (la qualita' sara' segnalata
    # a valle dal "pavimento di qualita'").
    order = np.argsort(distances)
    kept_end: list[int] = []
    kept_dist: list[float] = []
    for idx in order:
        d = distances[idx]
        if d > threshold_distance and len(kept_end) >= min_matches:
            break
        e = int(end_indices[idx])
        if any(abs(e - ke) < min_gap for ke in kept_end):
            continue  # troppo vicino a un match gia' tenuto -> stesso evento
        kept_end.append(e)
        kept_dist.append(float(d))
        if len(kept_end) >= max_matches:
            break

    kept_end_arr = np.array(kept_end, dtype=int)
    kept_dist_arr = np.array(kept_dist)
    sims = _similarity_from_distance(kept_dist_arr, n_dims)

    return MatchResult(
        end_indices=kept_end_arr,
        similarities=sims,
        distances=kept_dist_arr,
        n_candidates=n_candidates,
        threshold_distance=threshold_distance,
        query_vector=query,
    )
