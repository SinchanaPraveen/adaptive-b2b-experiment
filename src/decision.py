# src/decision.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.metrics import (
    ab_proportion_metric,
    guardrail_summary,
    segment_effects,
    sequential_daily_report,
)


@dataclass(frozen=True)
class DecisionConfig:
    # Sequential testing params
    alpha: float = 0.01
    min_days: int = 5
    min_total_sessions: int = 20000

    # Practical significance thresholds (percentage points)
    min_uplift_pp: float = 0.30          # overall practical uplift threshold
    min_uplift_seg_pp: float = 0.60      # segment practical uplift threshold

    # Segment sufficiency
    min_segment_sessions: int = 12000    # total (A+B) in segment

    # Guardrails
    max_load_delta_ms: float = 80.0
    max_bounce_uplift_pp: float = 1.0

    # Metrics
    primary_metric: str = "explanation_click"
    secondary_metric: str = "action_taken"


def _guardrails_ok(df: pd.DataFrame, cfg: DecisionConfig) -> Tuple[bool, Dict[str, float], List[str]]:
    g = guardrail_summary(df)
    issues: List[str] = []
    if g["load_delta_ms"] > cfg.max_load_delta_ms:
        issues.append(
            f"Load time increase too high: Δ{g['load_delta_ms']:+.1f}ms (limit {cfg.max_load_delta_ms:.1f}ms)"
        )
    if g["bounce_uplift_pp"] > cfg.max_bounce_uplift_pp:
        issues.append(
            f"Bounce uplift too high: {g['bounce_uplift_pp']:+.2f}pp (limit {cfg.max_bounce_uplift_pp:.2f}pp)"
        )
    return (len(issues) == 0), g, issues


def _format_segment_row(row: pd.Series) -> str:
    # Example: "persona=manager, usage_level=casual"
    keys = []
    for c in ["persona", "usage_level"]:
        if c in row:
            keys.append(f"{c}={row[c]}")
    return ", ".join(keys)


def decide(
    df: pd.DataFrame,
    cfg: DecisionConfig,
) -> Dict[str, object]:
    """
    Returns a structured decision dict:
      decision: LAUNCH / KILL / CONTINUE / TARGETED_ROLLOUT / HOLD_GUARDRAILS
      rationale: list[str]
      stop_day: Optional[date]
      daily_report: pd.DataFrame
      overall_primary: ABResult-like dict
      overall_secondary: ABResult-like dict
      guardrails: dict
      segments: pd.DataFrame
      target_segments: list[str]
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the dataset.")
    df["date"] = pd.to_datetime(df["date"])

    # Guardrails (evaluated on all observed data)
    ok_guard, g, guard_issues = _guardrails_ok(df, cfg)

    # Overall results
    prim = ab_proportion_metric(df, cfg.primary_metric)
    sec = ab_proportion_metric(df, cfg.secondary_metric)

    # Daily sequential monitoring on PRIMARY metric (you can switch to secondary later if you want)
    daily = sequential_daily_report(
        df,
        metric_col=cfg.primary_metric,
        alpha=cfg.alpha,
        min_days=cfg.min_days,
        min_total_sessions=cfg.min_total_sessions,
    )

    # Segment effects on SECONDARY metric (decision impact metric)
    seg = segment_effects(df, cfg.secondary_metric)

    rationale: List[str] = []
    target_segments: List[str] = []

    # Early stopping: find first eligible day where p < alpha
    stop_row = None
    eligible_rows = daily[daily["eligible"] & daily["signal_at_alpha"]]
    if len(eligible_rows) > 0:
        stop_row = eligible_rows.iloc[0]  # first trigger

    # Decide based on early stop or final day
    if stop_row is not None:
        stop_day = stop_row["day"]
        uplift = float(stop_row["uplift_pp"])
        pval = float(stop_row["p_value"])
        rationale.append(
            f"Early signal on {stop_day}: primary uplift {uplift:+.2f}pp with p={pval:.4g} (alpha={cfg.alpha})."
        )

        if not ok_guard:
            rationale.append("Guardrails violated; do not launch as-is.")
            rationale.extend(guard_issues)
            decision = "HOLD_GUARDRAILS"
        else:
            if uplift >= cfg.min_uplift_pp:
                decision = "LAUNCH"
                rationale.append(
                    f"Uplift exceeds practical threshold ({cfg.min_uplift_pp:.2f}pp) and guardrails are OK."
                )
            elif uplift <= -cfg.min_uplift_pp:
                decision = "KILL"
                rationale.append(
                    f"Negative uplift exceeds practical threshold (-{cfg.min_uplift_pp:.2f}pp); recommend rollback."
                )
            else:
                decision = "CONTINUE"
                rationale.append(
                    f"Statistically significant but below practical threshold ({cfg.min_uplift_pp:.2f}pp); continue to assess."
                )
    else:
        # No early stop: evaluate at end
        stop_day = None
        rationale.append(
            f"No early stopping trigger found (alpha={cfg.alpha}, min_days={cfg.min_days}, min_sessions={cfg.min_total_sessions})."
        )

        if not ok_guard:
            decision = "HOLD_GUARDRAILS"
            rationale.append("Guardrails violated; optimize performance and re-test.")
            rationale.extend(guard_issues)
        else:
            # Use end-of-test primary stats
            end = daily.iloc[-1]
            uplift = float(end["uplift_pp"])
            pval = float(end["p_value"])

            if (pval < cfg.alpha) and (uplift >= cfg.min_uplift_pp):
                decision = "LAUNCH"
                rationale.append(
                    f"At end: primary uplift {uplift:+.2f}pp with p={pval:.4g}. Meets statistical + practical thresholds."
                )
            elif (pval < cfg.alpha) and (uplift <= -cfg.min_uplift_pp):
                decision = "KILL"
                rationale.append(
                    f"At end: primary uplift {uplift:+.2f}pp with p={pval:.4g}. Negative impact; recommend rollback."
                )
            else:
                # Consider targeted rollout based on SECONDARY segment effects
                # Find segments with strong positive uplift and statistically significant
                seg_candidates = seg[
                    (seg["n_total"] >= cfg.min_segment_sessions)
                    & (seg["p_value"] < cfg.alpha)
                    & (seg["uplift_pp"] >= cfg.min_uplift_seg_pp)
                ].copy()

                if len(seg_candidates) > 0:
                    decision = "TARGETED_ROLLOUT"
                    for _, r in seg_candidates.iterrows():
                        target_segments.append(_format_segment_row(r))
                    rationale.append(
                        "Overall results are inconclusive, but some segments show strong positive action uplift."
                    )
                    rationale.append(
                        f"Recommend targeted rollout to: {', '.join(target_segments)}"
                    )
                else:
                    decision = "CONTINUE"
                    rationale.append(
                        "Overall results are inconclusive and no segments meet targeted rollout thresholds."
                    )

    return {
        "decision": decision,
        "rationale": rationale,
        "stop_day": stop_day,
        "daily_report": daily,
        "overall_primary": prim,
        "overall_secondary": sec,
        "guardrails": g,
        "segments": seg,
        "target_segments": target_segments,
    }
