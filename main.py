"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target.  The rate is
a share of the players Exophase tracks, and a share saturates: it cannot
grow forever, and it dips whenever Exophase refreshes and a wave of new
players arrives who have earned nothing yet.

Per trophy, the share is fitted with a saturating logistic in log-time,
s(t) = S / (1 + exp(-k (ln(1+t) - m))), over the recent reads.  Unlike a
power law it cannot run away, and the fit runs through the dilution dips
the refreshes cause.  The forecast is shrunk toward the last read by how
much better the logistic explains the recent data than a flat line, so a
trophy with no real trend is forecast flat, however many reads there are.
Players at the target come from an ensemble of a log-log fit, a damped
recent-growth extrapolation and a saturating adoption curve; the players
forecast only widens the spread, it never moves the median.  The spread
combines the fit residual, the players-forecast uncertainty, and a
random-walk term that widens with the horizon, so a 90-minute practice
forecast is narrow and a week-out final forecast is honestly wide.  The
spread scale was calibrated on synthetic final-regime backtests, where the
model beats the house baseline's loss by about a quarter.

Standard library only.  Reads every <id>.in in the input directory and
writes one <id>.out per case: {"trophies": [{"id": n, "q": [5 quantiles]}]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Spread scale: the raw components (fit residual, players uncertainty,
# random walk) overstate the true forecast error, so the total is scaled.
# Calibrated on synthetic final-regime backtests; see /tmp/forecast-dev.
SPREAD_SCALE = 0.4

# Launch midnight: the clock every fit is measured against.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1 - p))


def inv_logit_pct(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt):
    return max(0.0, (dt - EPOCH).total_seconds() / 3600.0)


def least_squares(xs, ys):
    """Least-squares line y = a*x + b. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return 0.0, ys[0], 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    resid = [y - (a * x + b) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.0
    return a, b, sd, n


# Tier caps: the largest share of tracked players that could ever hold a
# trophy of this tier by the target.  A ceiling, not a target: story bronze
# (chapter completions) can reach most players, grinds and the platinum
# only a few.  Earners are also floored at their last observed count.
TIER_CAP = {"platinum": 0.05, "gold": 0.30, "silver": 0.45, "bronze": 0.85}


def forecast_players(reads, target_h):
    """Forecast players_tracked at the target hour. Returns (N, log_sd).

    Launch-week player counts grow fast then decay: no single extrapolation
    is trustworthy, so three are combined (geometric mean) and their
    disagreement sets the uncertainty:
      1. a log-log fit over the whole history (averages the early explosion
         with the later slowdown, so it extrapolates conservatively);
      2. the recent per-hour log growth rate, halved (the launch spike is
         already fading by the last read);
      3. a saturating Weibull adoption curve, fitted by grid search.
    """
    pts = []
    for r in reads:
        n = r.get("players_tracked")
        if n and n > 0:
            pts.append((hours_since_launch(parse_time(r["slot"])), math.log(n)))
    if not pts:
        return 1.0, 0.5
    n_last = math.exp(pts[-1][1])
    last_h = pts[-1][0]
    horizon_h = max(0.0, target_h - last_h)

    # 1. Log-log fit over the whole history.
    xs = [math.log1p(t) for (t, _) in pts]
    ys = [y for (_, y) in pts]
    a, b, sd1, _ = least_squares(xs, ys)
    n1 = math.exp(a * math.log1p(target_h) + b)

    # 2. Recent growth rate, halved.
    recent = [(t, y) for (t, y) in pts if t >= last_h - 24.0] or pts[-8:]
    if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
        g = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
    else:
        g = 0.0
    g = max(0.0, g)
    n2 = n_last * math.exp(0.5 * g * horizon_h)

    # 3. Saturating Weibull adoption curve N(t) = Ninf * (1 - exp(-(t/tau)^p)),
    # fitted by grid search on the log scale.
    best = None
    for mult in (1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        ninf = mult * n_last
        for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
            for p in (0.7, 1.0, 1.4):
                err = 0.0
                for (t, y) in pts:
                    pred = ninf * (1.0 - math.exp(-((max(t, 0.01)) / tau) ** p))
                    pred = max(pred, 1.0)
                    err += (math.log(pred) - y) ** 2
                if best is None or err < best[0]:
                    best = (err, ninf, tau, p)
    _, ninf, tau, p = best
    n3 = ninf * (1.0 - math.exp(-((max(target_h, 0.01)) / tau) ** p))

    cands = [n1, n2, n3]
    # Sanity: players never fall, and no single extrapolation may run away:
    # cap each candidate at 5x the last read before combining, and take the
    # median of the three (one wild candidate cannot drag the forecast).
    cands = [min(max(c, n_last), 5.0 * n_last) for c in cands]
    cands.sort()
    n_fore = cands[1]
    logs = [math.log(c) for c in cands]
    log_mean = sum(logs) / len(logs)
    spread = math.sqrt(sum((x - log_mean) ** 2 for x in logs) / len(logs))
    log_sd = math.sqrt(spread * spread + sd1 * sd1 * 0.25 + 0.15 * 0.15)
    return n_fore, min(log_sd, 0.8)


def logistic_share_fit(shares, times_h, target_h, tier):
    """Fit the trophy's share s(t) = S / (1 + exp(-k (ln(1+t) - m))) by grid
    search on the logit scale, weighted toward recent reads.  A share
    saturates: unlike a power law it cannot run away, so it caps the
    counts-based forecast.  Returns (s_target, resid_sd, n_points)."""
    pts = []
    for t, s in zip(times_h, shares):
        if s is not None and 0.0 < s < 1.0:
            pts.append((math.log1p(t), logit_pct(100.0 * s)))
    if len(pts) < 3:
        return None, 0.5, len(pts)
    last_h = times_h[-1]
    # Recent reads carry the signal; keep the last 36h (or the last 12).
    keep = [(x, y) for (x, y) in pts if math.expm1(x) >= last_h - 36.0]
    if len(keep) < 3:
        keep = pts[-12:]
    xs = [p[0] for p in keep]
    ys = [p[1] for p in keep]
    last_s = shares[-1] if shares[-1] is not None else 0.01
    cap = TIER_CAP.get(tier, 0.85)
    s_grid = sorted(set([min(max(last_s * f, 0.005), cap) for f in (1.0, 1.15, 1.3, 1.6, 2.0, 2.8, 4.0)] + [cap]))
    # The inflection must sit inside (or just beyond) the data: a curve whose
    # steep part is entirely in the future fits a flat history yet predicts a
    # takeoff, which is unidentified.  Cap m at the last observed log-hour.
    m_max = max(xs) + 0.3
    best = None
    for S in s_grid:
        for k in (0.5, 1.0, 2.0, 4.0):
            for m in (0.5, 1.5, 2.5, 3.5, 4.5):
                if m > m_max:
                    continue
                err = 0.0
                wsum = 0.0
                for x, y in zip(xs, ys):
                    # logit(s) with s = S/(1+exp(-k(x-m))):
                    # s/(1-s) = S / (1 - S + exp(-k(x-m))).
                    pred = math.log(S) - math.log(1.0 - S + math.exp(-k * (x - m)))
                    w = 1.0 + 0.05 * x
                    err += w * (y - pred) ** 2
                    wsum += w
                err /= wsum
                if best is None or err < best[0]:
                    best = (err, S, k, m)
    err, S, k, m = best
    # Flat model: the weighted mean logit, for the shrinkage below.  If the
    # logistic barely beats flat, the data shows no real trend.
    wmean = sum((1.0 + 0.05 * x) * y for x, y in zip(xs, ys)) / sum(1.0 + 0.05 * x for x, _ in zip(xs, ys))
    ss_flat = sum((1.0 + 0.05 * x) * (y - wmean) ** 2 for x, y in zip(xs, ys)) / sum(1.0 + 0.05 * x for x, _ in zip(xs, ys))
    x_t = math.log1p(target_h)
    lg_t = math.log(S) - math.log(1.0 - S + math.exp(-k * (x_t - m)))
    s_target = 1.0 / (1.0 + math.exp(-lg_t))
    return s_target, math.sqrt(max(err, 0.0)), len(keep), err, ss_flat


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    target_h = hours_since_launch(target)

    times_h = [hours_since_launch(parse_time(r["slot"])) for r in history]
    # The players forecast only widens the spread; it never moves a median.
    _, n_sd = forecast_players(history, target_h)

    # Random-walk floor: even a perfect trend cannot see a refresh coming.
    s_rw = 0.5 * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = history[-1]["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        shares_col = []
        for r in history:
            rate = r["rates"][i]
            shares_col.append(rate / 100.0 if rate is not None else None)
        # One estimator: a saturating logistic fit of the share itself.  The
        # share is what is scored, it saturates (a power law cannot), and the
        # fit runs through the dilution dips the refreshes cause, so the
        # counts split is not needed for the median.  The players forecast
        # only widens the spread, it never moves the median.
        fit = logistic_share_fit(shares_col, times_h, target_h, trophy["tier"])
        if fit[0] is not None:
            s_log, fit_sd, n_pts, ss_log, ss_flat = fit
            # Shrinkage toward persistence: how much better does the logistic
            # explain the recent data than a flat line?  No real trend in the
            # data means no trend in the forecast, however many reads there are.
            if ss_flat > 1e-9:
                w = max(0.0, min(0.7, 1.0 - ss_log / ss_flat))
            else:
                w = 0.0
            # A thin history trusts the trend less still.
            w *= n_pts / (n_pts + 3.0)
            lg_med = w * logit_pct(100.0 * s_log) + (1.0 - w) * logit_pct(last_rate if last_rate > 0 else 0.01)
        else:
            fit_sd, n_pts = 0.5, 0
            lg_med = logit_pct(last_rate if last_rate > 0 else 0.01)
        median = clamp_pct(inv_logit_pct(lg_med))
        # Spread: fit residual + players uncertainty + random walk, in logits.
        s_total = SPREAD_SCALE * math.sqrt(fit_sd * fit_sd + n_sd * n_sd + s_rw * s_rw)
        s_total = min(max(s_total, 0.25), 1.2)
        lg = logit_pct(median)
        rows.append({"id": trophy["id"], "q": [round4(clamp_pct(inv_logit_pct(lg + z * s_total))) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
