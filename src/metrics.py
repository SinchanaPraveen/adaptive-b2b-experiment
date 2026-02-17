# src/metrics.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class ABResult:
    metric: str
    n_A: int
    n_B: int
    rate_A: float
    rate_B: float
    uplift_pp: float
    z: float
    p_value: float
    ci_low_pp: float
    ci_high_pp: float


def _two_proportion_ztest(
    x_A: int, n_A: int, x_B: int, n_B: int
) -> Tuple[float, float]:
    """
    Two-sided z-test for difference in proportions.
    Returns: (z, p_value)
    """
    if n_A == 0 or n_B == 0:
        return np.nan, np.nan

    pA = x_A / n_A
    pB = x_B / n_B
    p_pool = (x_A + x_B) / (n_A + n_B)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))
    if se == 0:
        return np.nan, np.nan
    z = (pB - pA) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def _diff_prop_ci_95(
    x_A: int, n_A: int, x_B: int, n_B: int
) -> Tuple[float, float]:
    """
    95% CI for (pB - pA) using normal approximation (unpooled SE).
    Returns bounds in proportion units (not percent).
    """
    if n_A == 0 or n_B == 0:
        return np.nan, np.nan
    pA = x_A / n_A
    pB = x_B / n_B
    se = np.sqrt(pA * (1 - pA) / n_A + pB * (1 - pB) / n_B)
    z = 1.96
    lo = (pB - pA) - z * se
    hi = (pB - pA) + z * se
    return float(lo), float(hi)


def ab_proportion_metric(df: pd.DataFrame, metric_col: str) -> ABResult:
    """
    Compute A/B stats for a binary metric column (0/1).
    """
    dfA = df[df["variant"] == "A"]
    dfB = df[df["variant"] == "B"]

    nA = int(len(dfA))
    nB = int(len(dfB))
    xA = int(dfA[metric_col].sum())
    xB = int(dfB[metric_col].sum())

    rateA = xA / nA if nA else np.nan
    rateB = xB / nB if nB else np.nan
    uplift = (rateB - rateA) * 100.0  # percentage points

    z, p = _two_proportion_ztest(xA, nA, xB, nB)
    lo, hi = _diff_prop_ci_95(xA, nA, xB, nB)
    return ABResult(
        metric=metric_col,
        n_A=nA,
        n_B=nB,
        rate_A=float(rateA),
        rate_B=float(rateB),
        uplift_pp=float(uplift),
        z=float(z),
        p_value=float(p),
        ci_low_pp=float(lo * 100.0),
        ci_high_pp=float(hi * 100.0),
    )


def guardrail_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Guardrails: load_time_ms (mean) and bounce rate by variant.
    """
    out: Dict[str, float] = {}
    for v in ["A", "B"]:
        d = df[df["variant"] == v]
        out[f"load_mean_{v}"] = float(d["load_time_ms"].mean())
        out[f"bounce_rate_{v}"] = float(d["bounce"].mean())
    out["load_delta_ms"] = out["load_mean_B"] - out["load_mean_A"]
    out["bounce_uplift_pp"] = (out["bounce_rate_B"] - out["bounce_rate_A"]) * 100.0
    return out


def sequential_daily_report(
    df: pd.DataFrame,
    metric_col: str,
    alpha: float = 0.01,
    min_days: int = 5,
    min_total_sessions: int = 20000,
) -> pd.DataFrame:
    """
    Cumulative-by-day sequential monitoring for a binary metric.
    Early stopping is *not* implemented here (that will go in decision.py),
    but this report gives you everything needed.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    days = sorted(df["date"].unique())

    rows = []
    for i, day in enumerate(days, start=1):
        d = df[df["date"] <= day]
        res = ab_proportion_metric(d, metric_col)

        eligible = (i >= min_days) and ((res.n_A + res.n_B) >= min_total_sessions)
        rows.append(
            {
                "day": day,
                "days_observed": i,
                "eligible": eligible,
                "n_total": res.n_A + res.n_B,
                "rate_A": res.rate_A,
                "rate_B": res.rate_B,
                "uplift_pp": res.uplift_pp,
                "p_value": res.p_value,
                "ci_low_pp": res.ci_low_pp,
                "ci_high_pp": res.ci_high_pp,
                "signal_at_alpha": bool(eligible and (res.p_value < alpha)),
            }
        )

    return pd.DataFrame(rows)


def segment_effects(
    df: pd.DataFrame,
    metric_col: str,
    segment_cols: Tuple[str, ...] = ("persona", "usage_level"),
) -> pd.DataFrame:
    """
    A/B uplift + p-value per segment cell (e.g., manager/power).
    """
    rows = []
    grp = df.groupby(list(segment_cols), dropna=False)
    for keys, d in grp:
        res = ab_proportion_metric(d, metric_col)
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {segment_cols[i]: keys[i] for i in range(len(segment_cols))}
        row.update(
            {
                "n_total": res.n_A + res.n_B,
                "rate_A": res.rate_A,
                "rate_B": res.rate_B,
                "uplift_pp": res.uplift_pp,
                "p_value": res.p_value,
                "ci_low_pp": res.ci_low_pp,
                "ci_high_pp": res.ci_high_pp,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(by="uplift_pp", ascending=False)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sessions.csv")
    ap.add_argument("--metric", type=str, default="action_taken",
                    choices=["explanation_click", "action_taken", "bounce"])
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])

    print("\n=== Overall A/B (binary metric) ===")
    overall = ab_proportion_metric(df, args.metric)
    print(
        f"{overall.metric}: A={overall.rate_A:.4f} (n={overall.n_A:,}) | "
        f"B={overall.rate_B:.4f} (n={overall.n_B:,}) | "
        f"uplift={overall.uplift_pp:+.2f}pp | p={overall.p_value:.4g} | "
        f"95%CI=[{overall.ci_low_pp:+.2f}, {overall.ci_high_pp:+.2f}]pp"
    )

    print("\n=== Guardrails ===")
    g = guardrail_summary(df)
    print(
        f"Load mean A={g['load_mean_A']:.1f}ms, B={g['load_mean_B']:.1f}ms "
        f"(Δ={g['load_delta_ms']:+.1f}ms)"
    )
    print(
        f"Bounce rate A={g['bounce_rate_A']:.4f}, B={g['bounce_rate_B']:.4f} "
        f"(uplift={g['bounce_uplift_pp']:+.2f}pp)"
    )

    print("\n=== Sequential daily report (cumulative) ===")
    rep = sequential_daily_report(df, args.metric, alpha=args.alpha)
    print(rep.tail(7).to_string(index=False))

    print("\n=== Segment effects (persona x usage_level) ===")
    seg = segment_effects(df, args.metric)
    print(seg.to_string(index=False))


if __name__ == "__main__":
    main()
