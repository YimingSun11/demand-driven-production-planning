"""Prep-quantity rule backtest for the Stuttgart restaurant dataset.

A "prep rule" tells a kitchen manager how many portions of an item to prepare
each day. Each rule groups weekdays and uses one number per group: an empirical
quantile of that group's historical daily sales.

Rules
-----
A  one number for every day
W  two numbers: Sat+Sun vs. Mon-Fri (pre-specified "weekend" split)
F  two numbers: Fri+Sat vs. the other five days (pre-specified)
B  two numbers: the k busiest weekdays vs. the rest; k re-chosen each window on validation data
C  seven numbers, one per weekday
(S, Saturday vs. the rest, is available as a sensitivity check.)

All results replay prep quantities against historical *sales* (not demand).
Leftover and shortfall are simulated portions, not measured waste.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "stuttgart_daily_sales.csv"
ITEMS_EN = {
    "LAMM": "Lamb chops", "STEAK": "Steak", "HAEHNCHEN": "Chicken", "KOEFTE": "Meatballs",
    "CALAMARI": "Squid", "FISCH": "Fish", "GARNELEN": "Shrimp",
}
MAIN_ITEMS = ["LAMM", "STEAK", "HAEHNCHEN", "KOEFTE"]
RULES = ["A", "W", "F", "B", "C"]
# Quantile grid shared by every rule. q = 1.0 is the historical maximum, so no
# rule can ever prep more than the most it has seen sold on a day in its group.
Q_GRID = np.round(np.arange(0.50, 1.0 + 1e-9, 0.005), 3)
QMETHOD = "inverted_cdf"  # always an observed sales value; matches rounds 1-2

ROLL_START = "2014-11-29"
WINDOW = 28          # trading days per test window
VAL_DAYS = 56        # trading days used to pick the safety level


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["date"] = pd.to_datetime(d["DEMAND_DATE"], format="%m.%d.%Y")
    d["wd"] = d["date"].dt.dayofweek  # Mon=0 ... Sat=5, Sun=6
    return d.drop(columns="DEMAND_DATE").sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- grouping
def group_map(rule: str, y: np.ndarray, wd: np.ndarray, k: int | None = None) -> np.ndarray:
    """Return an array g[weekday] -> group id, using only the data passed in."""
    if rule == "A":
        return np.zeros(7, int)
    if rule == "W":
        return np.array([0, 0, 0, 0, 0, 1, 1])
    if rule == "F":  # pre-specified Fri+Sat vs. other days
        return np.array([0, 0, 0, 0, 1, 1, 0])
    if rule == "S":  # pre-specified Saturday vs. other days
        return np.array([0, 0, 0, 0, 0, 1, 0])
    if rule == "C":
        return np.arange(7)
    if rule == "B":
        means = np.array([y[wd == w].mean() for w in range(7)])
        busiest = np.argsort(-means)[:k]
        g = np.zeros(7, int)
        g[busiest] = 1
        return g
    raise ValueError(rule)


def fit_table(y, wd, gmap, q) -> np.ndarray:
    """Prep quantity per weekday: quantile q of each group's history."""
    table = np.empty(7)
    for gid in np.unique(gmap):
        days = np.where(gmap == gid)[0]
        table[days] = np.quantile(y[np.isin(wd, days)], q, method=QMETHOD)
    return table


def apply_prep(table, wd, buffer=0.0, batch=None) -> np.ndarray:
    p = table[wd] * (1.0 + buffer)
    if batch:
        p = np.ceil(p / batch) * batch
    else:
        p = np.ceil(p)  # portions are whole numbers
    return p


# ---------------------------------------------------------------- metrics
def metrics(prep, sold, wd) -> dict:
    left = np.maximum(prep - sold, 0)
    short = np.maximum(sold - prep, 0)
    sat = wd == 5
    return {
        "days": len(sold),
        "sold": sold.sum(),
        "served": np.minimum(prep, sold).sum(),
        "fill": np.minimum(prep, sold).sum() / sold.sum(),   # unit-weighted
        "leftover": left.sum(),
        "short": short.sum(),
        "prep": prep.sum(),
        "sat_days": sat.sum(),
        "sat_covered": (prep[sat] >= sold[sat]).sum(),
    }


def pooled(rows: list[dict]) -> dict:
    keys = ["days", "sold", "served", "leftover", "short", "prep", "sat_days", "sat_covered"]
    t = {k: sum(r[k] for r in rows) for k in keys}
    return {
        "fill": t["served"] / t["sold"],
        "leftover_per_day": t["leftover"] / t["days"],
        "short_per_day": t["short"] / t["days"],
        "prep_per_day": t["prep"] / t["days"],
        "sat_covered": t["sat_covered"] / t["sat_days"] if t["sat_days"] else np.nan,
        "days": t["days"],
    }


# ---------------------------------------------------------------- calibration
@dataclass
class Choice:
    q: float
    k: int | None
    reached: bool  # did any q on the grid hit the target on validation data?


def choose(rule, y_fit, wd_fit, y_val, wd_val, target, buffer=0.0, batch=None) -> Choice:
    """Pick the lowest quantile (and, for B, the grouping) that meets the target
    on the validation slice. Rules are fit on data *before* the validation slice."""
    ks = range(1, 7) if rule == "B" else [None]
    best = None
    for k in ks:
        g = group_map(rule, y_fit, wd_fit, k)
        chosen = None
        for q in Q_GRID:
            p = apply_prep(fit_table(y_fit, wd_fit, g, q), wd_val, buffer, batch)
            m = metrics(p, y_val, wd_val)
            if m["fill"] >= target:
                chosen = (q, m["leftover"], True)
                break
        if chosen is None:  # target unreachable on the grid -> use the cap
            q = Q_GRID[-1]
            p = apply_prep(fit_table(y_fit, wd_fit, g, q), wd_val, buffer, batch)
            chosen = (q, metrics(p, y_val, wd_val)["leftover"], False)
        cand = (not chosen[2], chosen[1], k, chosen[0])  # prefer reached, then less leftover
        if best is None or cand < best:
            best = cand
    return Choice(q=best[3], k=best[2], reached=not best[0])


def window_starts(d: pd.DataFrame, start=ROLL_START, window=WINDOW):
    i0 = int((d["date"] < start).sum())
    return list(range(i0, len(d), window))


def rolling(d, item, rule, target, val_days=VAL_DAYS, buffer=0.0, batch=None,
            history=None, start=ROLL_START, window=WINDOW):
    """Deployable rolling backtest. For each test window, only rows before it
    are used. Returns per-window metric rows (with chosen parameters).

    history: if set, refit on only the last `history` rows (else all prior rows).
    """
    y_all, wd_all = d[item].to_numpy(float), d["wd"].to_numpy()
    out = []
    for s in window_starts(d, start, window):
        e = min(s + window, len(d))
        v0 = s - val_days
        lo_fit = 0 if history is None else max(0, v0 - history)
        # The safety level is chosen WITHOUT the buffer; the buffer is then added on
        # top at deployment. (Choosing with the buffer would just cancel it out.)
        ch = choose(rule, y_all[lo_fit:v0], wd_all[lo_fit:v0],
                    y_all[v0:s], wd_all[v0:s], target, 0.0, batch)
        lo_ref = 0 if history is None else max(0, s - history)
        y_hist, wd_hist = y_all[lo_ref:s], wd_all[lo_ref:s]
        g = group_map(rule, y_hist, wd_hist, ch.k)
        table = fit_table(y_hist, wd_hist, g, ch.q)
        p = apply_prep(table, wd_all[s:e], buffer, batch)
        m = metrics(p, y_all[s:e], wd_all[s:e])
        m.update(window_start=d["date"].iloc[s].date(), q=ch.q, k=ch.k, reached_on_val=ch.reached)
        out.append(m)
    return out


def rolling_fixed_q(d, item, rule, q, batch=None, k_from_val=True, val_days=VAL_DAYS,
                    start=ROLL_START, window=WINDOW):
    """Same windows, but the quantile q is fixed in advance for every window.
    Used for the ex-post equal-fill-rate comparison. For B the grouping is still
    chosen per window from pre-window data (validation leftover at this q)."""
    y_all, wd_all = d[item].to_numpy(float), d["wd"].to_numpy()
    out = []
    for s in window_starts(d, start, window):
        e = min(s + window, len(d))
        k = None
        if rule == "B":
            v0 = s - val_days
            best = None
            for kk in range(1, 7):
                g = group_map("B", y_all[:v0], wd_all[:v0], kk)
                p = apply_prep(fit_table(y_all[:v0], wd_all[:v0], g, q), wd_all[v0:s], 0, batch)
                lv = metrics(p, y_all[v0:s], wd_all[v0:s])["leftover"]
                if best is None or lv < best[0]:
                    best = (lv, kk)
            k = best[1]
        g = group_map(rule, y_all[:s], wd_all[:s], k)
        p = apply_prep(fit_table(y_all[:s], wd_all[:s], g, q), wd_all[s:e], 0, batch)
        out.append(metrics(p, y_all[s:e], wd_all[s:e]))
    return out
