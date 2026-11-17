"""Reproduce every table and the chart: python3 run_all.py  (about 5 minutes)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from prep_sim import (MAIN_ITEMS, ITEMS_EN, RULES, load, rolling, pooled,  # noqa: E402
                      rolling_fixed_q, window_starts)
from expost import curve, at_fill  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
d = load()
TARGETS = [0.95, 0.96, 0.97, 0.98, 0.99]

# Calibration methods (all use only data before each test window)
METHODS = {
    "M1 quantile (56-day validation)": dict(val_days=56, buffer=0.0),
    "M2 quantile + 5% buffer": dict(val_days=56, buffer=0.05),
    "M3 quantile (112-day validation)": dict(val_days=112, buffer=0.0),
}
SENS = {
    "M2a quantile + 3% buffer": dict(val_days=56, buffer=0.03),
    "M2b quantile + 10% buffer": dict(val_days=56, buffer=0.10),
}


def summarise(rows, target):
    p = pooled(rows)
    full = [r for r in rows if r["days"] == 28]
    p["windows_below_target"] = sum(r["served"] / r["sold"] < target for r in full)
    p["windows"] = len(full)
    return p


# 1. Deployable backtest: every method x rule x item x target
recs = []
for mname, kw in {**METHODS, **SENS}.items():
    for item in MAIN_ITEMS:
        for rule in RULES:
            for t in TARGETS:
                if mname in SENS and t != 0.98:
                    continue
                rows = rolling(d, item, rule, t, **kw)
                s = summarise(rows, t)
                recs.append(dict(method=mname, item=ITEMS_EN[item], rule=rule, target=t, **s))
dep = pd.DataFrame(recs)
dep.to_csv(OUT / "deployable_backtest.csv", index=False)

# 1b. Batch-of-5 robustness for M1 at 98%
recs = []
for item in MAIN_ITEMS:
    for rule in RULES:
        s = summarise(rolling(d, item, rule, 0.98, batch=5), 0.98)
        recs.append(dict(item=ITEMS_EN[item], rule=rule, **s))
pd.DataFrame(recs).to_csv(OUT / "deployable_batch5_M1_98.csv", index=False)

# 2. Ex-post equal-fill-rate comparison (uses test data -> explanatory only)
recs = []
for item in MAIN_ITEMS:
    for rule in RULES:
        c = curve(d, item, rule)
        for t in [0.95, 0.97, 0.98, 0.99]:
            lv, sat = at_fill(c, t)
            recs.append(dict(item=ITEMS_EN[item], rule=rule, fill=t, leftover_per_day=lv, sat_covered=sat))
ep = pd.DataFrame(recs)
ep.to_csv(OUT / "expost_equal_fill.csv", index=False)

# 3. Ceiling check: fixed rule A, per-window fill at grid caps 0.99 and 1.0 (= historical max)
recs = []
for item in MAIN_ITEMS:
    for q in [0.99, 1.0]:
        for s, r in zip(window_starts(d), rolling_fixed_q(d, item, "A", q)):
            if r["days"] == 28:
                recs.append(dict(item=ITEMS_EN[item], cap_q=q, window_start=d["date"].iloc[s].date(),
                                 fill=r["served"] / r["sold"]))
cl = pd.DataFrame(recs)
cl.to_csv(OUT / "ceiling_check.csv", index=False)

# ---------------------------------------------------------------- printed summary
pd.set_option("display.width", 200)
m1 = dep[(dep.method.str.startswith("M1")) & (dep.target == 0.98)]
print("\nDEPLOYABLE (M1, target 98%)\n", m1[["item", "rule", "fill", "leftover_per_day",
      "sat_covered", "windows_below_target", "windows"]].round(3).to_string(index=False))

cal = dep[dep.target == 0.98].groupby(["method", "rule"]).agg(
    fill_min=("fill", "min"), fill_max=("fill", "max"), leftover=("leftover_per_day", "mean"),
    below=("windows_below_target", "sum"), windows=("windows", "sum")).reset_index()
print("\nCALIBRATION @98% (across 4 items)\n", cal.round(3).to_string(index=False))

wide = ep[ep.fill == 0.98].pivot(index="item", columns="rule", values="leftover_per_day")[RULES]
wide["F_vs_A"] = 1 - wide["F"] / wide["A"]
wide["B_vs_A"] = 1 - wide["B"] / wide["A"]
print("\nEX-POST @98% leftover/day\n", wide.round(3).to_string())
print("\nEX-POST F vs A range by target\n",
      ep.pivot_table(index=["fill", "item"], columns="rule", values="leftover_per_day")
      .assign(r=lambda x: 1 - x.F / x.A).groupby("fill").r.agg(["min", "max"]).round(3))
# 4. Deployable matched-fill comparison: interpolate each rule's deployable curve
#    (M1 + M2 points, targets 95-99%) at a common realized fill rate.
recs = []
pts = dep[dep.method.str.match("M[12] ")]
for item, g in pts.groupby("item"):
    for f in [0.965, 0.97, 0.975]:
        row = dict(item=item, fill=f)
        for rule in RULES:
            h = g[g.rule == rule].sort_values("fill")
            row[rule] = np.interp(f, h.fill, h.leftover_per_day) if h.fill.min() <= f <= h.fill.max() else np.nan
        recs.append(row)
mf = pd.DataFrame(recs)
for r in ["W", "F", "B", "C"]:
    mf[f"{r}_vs_A"] = 1 - mf[r] / mf["A"]
mf.to_csv(OUT / "deployable_matched_fill.csv", index=False)
print("\nDEPLOYABLE MATCHED FILL\n", mf.round(3).to_string(index=False))
# 5. Same preset target, no matching: F vs A in every deployable run (M1, M2 x 5 targets)
pt = dep[dep.method.str.match("M[12] ")].pivot_table(
    index=["method", "target", "item"], columns="rule",
    values=["fill", "leftover_per_day", "short_per_day"])
pre = pd.DataFrame({
    "fill_A": pt[("fill", "A")], "fill_F": pt[("fill", "F")],
    "fill_diff_pp": (pt[("fill", "F")] - pt[("fill", "A")]) * 100,
    "short_diff_per_day": pt[("short_per_day", "F")] - pt[("short_per_day", "A")],
    "leftover_A": pt[("leftover_per_day", "A")], "leftover_F": pt[("leftover_per_day", "F")],
    "F_vs_A": 1 - pt[("leftover_per_day", "F")] / pt[("leftover_per_day", "A")],
}).reset_index()
pre.to_csv(OUT / "deployable_same_target_F_vs_A.csv", index=False)
print("\nSAME PRESET TARGET, F vs A (40 runs): fill diff %.2f to %.2f pp; leftover cut %.1f%% to %.1f%% (median %.1f%%)" % (
    pre.fill_diff_pp.min(), pre.fill_diff_pp.max(), 100 * pre.F_vs_A.min(), 100 * pre.F_vs_A.max(), 100 * pre.F_vs_A.median()))

# 6. Buffer tradeoff: extra leftover per portion of shortfall avoided (F, 98%, M2 vs M1)
b = dep[(dep.rule == "F") & (dep.target == 0.98) & dep.method.str.match("M[12] ")].pivot_table(
    index="item", columns="method", values=["leftover_per_day", "short_per_day"])
m1, m2 = "M1 quantile (56-day validation)", "M2 quantile + 5% buffer"
bt = pd.DataFrame({
    "extra_leftover": b[("leftover_per_day", m2)] - b[("leftover_per_day", m1)],
    "shortfall_avoided": b[("short_per_day", m1)] - b[("short_per_day", m2)]})
bt["leftover_per_shortfall_avoided"] = bt.extra_leftover / bt.shortfall_avoided
bt.to_csv(OUT / "buffer_tradeoff.csv")
print("\nBUFFER TRADEOFF (F, 98%)\n", bt.round(2).to_string())

# 7. Small dishes (3.8-10.5 sold/day): M1, 98%
from prep_sim import ITEMS_EN as _EN
recs = []
for item in ["CALAMARI", "FISCH", "GARNELEN"]:
    for rule in ["A", "F", "C"]:
        s_ = summarise(rolling(d, item, rule, 0.98), 0.98)
        recs.append(dict(item=_EN[item], rule=rule, **s_))
sm = pd.DataFrame(recs)
sm.to_csv(OUT / "small_items_M1_98.csv", index=False)
print("\nSMALL DISHES (M1, 98%)\n", sm[["item", "rule", "fill", "leftover_per_day"]].round(3).to_string(index=False))
print("\nCEILING: windows below 98%\n", cl[cl.fill < 0.98].round(3).to_string(index=False))
