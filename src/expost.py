"""Ex-post comparison: same fixed quantile in every window, then interpolate to
the quantile at which the pooled test fill rate equals the target. Uses test
data to choose the level -> explanatory only, not achievable in advance."""
import numpy as np
from prep_sim import rolling_fixed_q, pooled, Q_GRID

def curve(d, item, rule, batch=None, grid=Q_GRID):
    pts = []
    for q in grid:
        p = pooled(rolling_fixed_q(d, item, rule, q, batch))
        pts.append((q, p["fill"], p["leftover_per_day"], p["sat_covered"]))
    return np.array(pts)

def at_fill(c, target):
    """Interpolate leftover and Saturday coverage at a pooled fill rate."""
    f, lv, sat = c[:, 1], c[:, 2], c[:, 3]
    order = np.argsort(f)
    f, lv, sat = f[order], lv[order], sat[order]
    if target > f.max():
        return np.nan, np.nan
    return float(np.interp(target, f, lv)), float(np.interp(target, f, sat))
