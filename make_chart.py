"""Chart: deployable tradeoff curves (realized fill rate vs. leftover), method M1.
Run after run_all.py:  python3 make_chart.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent
dep = pd.read_csv(ROOT / "results" / "deployable_backtest.csv")
dep = dep[dep.method.str.startswith("M1")]

STYLE = {  # validated categorical slots 1-4, plus marker shape as a second cue
    "A": ("#2a78d6", "o", "A  one number"),
    "W": ("#eb6834", "s", "W  Sat+Sun / weekdays"),
    "F": ("#1baf7a", "D", "F  Fri+Sat / other days"),
    "C": ("#eda100", "^", "C  seven numbers"),
}
INK, MUTED, GRID, SURF = "#22211e", "#6b6a64", "#e6e5e0", "#fcfcfb"
items = ["Lamb chops", "Steak", "Chicken", "Meatballs"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), facecolor=SURF)
for ax, item in zip(axes, items):
    ax.set_facecolor(SURF)
    g = dep[dep.item == item]
    ends = []
    for rule, (col, mk, _) in STYLE.items():
        h = g[g.rule == rule].sort_values("target")
        x, y = h.fill * 100, h.leftover_per_day
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5, zorder=3)
        ends.append([y.iloc[-1], x.iloc[-1], y.iloc[-1], rule])
    # spread end labels vertically so none overlap (min gap = 7% of the y range)
    top = max(e[0] for e in ends)
    gap = 0.07 * top * 1.1
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][2] = max(ends[i][2], ends[i - 1][2] + gap)
    for _, x_end, y_lab, rule in ends:
        ax.annotate(rule, (x_end, y_lab), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold")
    ax.axvline(98, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_title(item, loc="left", fontsize=10, fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.set_xlabel("Realized fill rate (%)")
    ax.set_ylim(bottom=0)
axes[0].set_ylabel("Simulated leftover (portions/day)")
handles = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=6, label=lab) for c, m, lab in STYLE.values()]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Deployable backtest: lower-right is better. Each point = one target setting (95-99%)",
             y=0.9, fontsize=9, color=MUTED)
fig.text(0.01, 0.01, "Rolling 28-day windows, Dec 2014-Nov 2015; rules set only from earlier data. "
         "Dashed line = 98%. Sales replay, not measured waste.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.86))
fig.savefig(ROOT / "results" / "tradeoff_chart.png", dpi=180, facecolor=SURF)
print("saved")
