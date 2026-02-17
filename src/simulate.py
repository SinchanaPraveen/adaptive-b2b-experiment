from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimConfig:
    days: int = 21
    sessions_per_day: int = 8000
    seed: int = 42

    # Traffic and population mix
    p_treatment: float = 0.50
    p_manager: float = 0.35
    p_power_user: float = 0.30

    # Baseline rates (control, variant A)
    baseline_explanation_click_rate: float = 0.06
    baseline_action_rate: float = 0.09
    baseline_bounce_rate: float = 0.18
    baseline_load_time_ms: float = 520.0
    load_time_sd_ms: float = 90.0

    # Natural segment offsets for action_taken
    action_manager_offset: float = 0.010
    action_power_user_offset: float = 0.015

    # Treatment effects (variant B)
    lift_explanation_click_rate: float = 0.030
    lift_action_manager_casual: float = 0.020
    lift_action_manager_power: float = 0.010
    lift_action_analyst_casual: float = 0.008
    lift_action_analyst_power: float = 0.003

    # Guardrail impacts in treatment
    lift_bounce_rate: float = 0.003
    lift_load_time_ms: float = 35.0


def _clip_prob(values: np.ndarray | float) -> np.ndarray:
    return np.clip(values, 1e-6, 1 - 1e-6)


def simulate_sessions(cfg: SimConfig, start: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    if start is None:
        start = date.today() - timedelta(days=cfg.days - 1)

    frames: list[pd.DataFrame] = []
    next_session_id = 1

    for day_idx in range(cfg.days):
        day = start + timedelta(days=day_idx)
        n = cfg.sessions_per_day

        is_treatment = rng.random(n) < cfg.p_treatment
        variant = np.where(is_treatment, "B", "A")

        is_manager = rng.random(n) < cfg.p_manager
        persona = np.where(is_manager, "manager", "analyst")

        is_power = rng.random(n) < cfg.p_power_user
        usage_level = np.where(is_power, "power", "casual")

        p_click = np.full(n, cfg.baseline_explanation_click_rate, dtype=float)
        p_click[is_treatment] += cfg.lift_explanation_click_rate
        explanation_click = rng.binomial(1, _clip_prob(p_click))

        p_action = np.full(n, cfg.baseline_action_rate, dtype=float)
        p_action[is_manager] += cfg.action_manager_offset
        p_action[is_power] += cfg.action_power_user_offset

        lift_action = np.zeros(n, dtype=float)
        mask_manager_casual = is_manager & ~is_power
        mask_manager_power = is_manager & is_power
        mask_analyst_casual = ~is_manager & ~is_power
        mask_analyst_power = ~is_manager & is_power

        lift_action[mask_manager_casual] = cfg.lift_action_manager_casual
        lift_action[mask_manager_power] = cfg.lift_action_manager_power
        lift_action[mask_analyst_casual] = cfg.lift_action_analyst_casual
        lift_action[mask_analyst_power] = cfg.lift_action_analyst_power
        p_action = p_action + lift_action * is_treatment.astype(float)
        action_taken = rng.binomial(1, _clip_prob(p_action))

        p_bounce = np.full(n, cfg.baseline_bounce_rate, dtype=float)
        p_bounce[is_treatment] += cfg.lift_bounce_rate
        bounce = rng.binomial(1, _clip_prob(p_bounce))

        load_time_ms = rng.normal(cfg.baseline_load_time_ms, cfg.load_time_sd_ms, size=n)
        load_time_ms += is_treatment.astype(float) * cfg.lift_load_time_ms
        load_time_ms = np.clip(load_time_ms, 100, 5000).round().astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime([day] * n),
                    "session_id": np.arange(next_session_id, next_session_id + n),
                    "variant": variant,
                    "persona": persona,
                    "usage_level": usage_level,
                    "explanation_click": explanation_click.astype(int),
                    "action_taken": action_taken.astype(int),
                    "bounce": bounce.astype(int),
                    "load_time_ms": load_time_ms,
                }
            )
        )
        next_session_id += n

    return pd.concat(frames, ignore_index=True)
