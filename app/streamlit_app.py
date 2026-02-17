# app/streamlit_app.py
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

# Make sure imports work when running: streamlit run app/streamlit_app.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulate import SimConfig, simulate_sessions
from src.decision import DecisionConfig, decide


st.set_page_config(page_title="Adaptive B2B A/B Testing", layout="wide")

st.title("Adaptive A/B Testing: Inline Explanations (B2B Analytics Dashboard)")
st.caption(
    "Sequential monitoring + early stopping + segment impact. Built for DS/DA-style product experimentation."
)

with st.sidebar:
    st.header("Data")
    use_sim = st.toggle("Use simulated data", value=True)

    if use_sim:
        st.subheader("Simulation settings")
        days = st.slider("Days", 7, 45, 21, 1)
        sessions_per_day = st.slider("Sessions per day", 1000, 20000, 8000, 500)
        seed = st.number_input("Seed", min_value=1, max_value=999999, value=42, step=1)

        cfg_sim = SimConfig(days=int(days), sessions_per_day=int(sessions_per_day), seed=int(seed))
        df = simulate_sessions(cfg_sim, start=date.today())
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to proceed, or switch back to simulated data.")
            st.stop()
        df = pd.read_csv(uploaded, parse_dates=["date"])

    st.header("Decision settings")
    primary = st.selectbox("Primary metric", ["explanation_click", "action_taken"], index=0)
    secondary = st.selectbox("Secondary metric", ["action_taken", "explanation_click"], index=0)

    alpha = st.select_slider("Alpha (significance)", options=[0.10, 0.05, 0.02, 0.01, 0.005], value=0.01)
    min_days = st.slider("Min days before stopping", 1, 14, 5, 1)
    min_sessions = st.slider("Min total sessions before stopping", 0, 100000, 20000, 1000)

    st.subheader("Practical thresholds (pp)")
    min_uplift_pp = st.slider("Overall min uplift (pp)", 0.0, 2.0, 0.30, 0.05)
    min_uplift_seg_pp = st.slider("Segment min uplift (pp)", 0.0, 3.0, 0.60, 0.05)
    min_seg_sessions = st.slider("Min segment sessions (A+B)", 1000, 80000, 12000, 1000)

    st.subheader("Guardrails")
    max_load_delta = st.slider("Max load time delta (ms)", 0, 300, 80, 5)
    max_bounce_pp = st.slider("Max bounce uplift (pp)", 0.0, 5.0, 1.0, 0.1)

cfg_dec = DecisionConfig(
    alpha=float(alpha),
    min_days=int(min_days),
    min_total_sessions=int(min_sessions),
    min_uplift_pp=float(min_uplift_pp),
    min_uplift_seg_pp=float(min_uplift_seg_pp),
    min_segment_sessions=int(min_seg_sessions),
    max_load_delta_ms=float(max_load_delta),
    max_bounce_uplift_pp=float(max_bounce_pp),
    primary_metric=primary,
    secondary_metric=secondary,
)

result = decide(df, cfg_dec)

# --- Top Summary ---
col1, col2, col3, col4 = st.columns(4)

prim = result["overall_primary"]
sec = result["overall_secondary"]
g = result["guardrails"]

col1.metric("Decision", result["decision"])
col2.metric("Primary uplift (pp)", f"{prim.uplift_pp:+.2f}", help=f"p={prim.p_value:.4g}")
col3.metric("Secondary uplift (pp)", f"{sec.uplift_pp:+.2f}", help=f"p={sec.p_value:.4g}")
col4.metric("Load Δ (ms)", f"{g['load_delta_ms']:+.1f}", help=f"Bounce uplift {g['bounce_uplift_pp']:+.2f}pp")

st.subheader("Decision rationale")
for line in result["rationale"]:
    st.write("• " + line)

if result["decision"] == "TARGETED_ROLLOUT":
    st.success("Target segments: " + ", ".join(result["target_segments"]))

st.divider()

# --- Daily charts ---
daily = result["daily_report"].copy()
daily["day"] = pd.to_datetime(daily["day"])

left, right = st.columns(2)

with left:
    st.subheader("Cumulative uplift over time (primary)")
    st.line_chart(daily.set_index("day")[["uplift_pp"]])

with right:
    st.subheader("Cumulative p-value over time (primary)")
    st.line_chart(daily.set_index("day")[["p_value"]])

st.caption("Tip: if p-value drops below alpha after min-days and min-sessions, early stopping can trigger.")

st.divider()

# --- Segment effects table ---
st.subheader("Segment effects (secondary metric)")
st.caption("Sorted by uplift (pp). Use this to justify targeted rollouts.")
seg = result["segments"].copy()
st.dataframe(seg, use_container_width=True)

st.divider()

# --- Raw data preview ---
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)
