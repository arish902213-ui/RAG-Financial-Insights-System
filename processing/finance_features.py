from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class OHLCAnalyticsResult:
    frame: pd.DataFrame
    metrics: Dict[str, object]


def _find_case_insensitive_column(df: pd.DataFrame, target: str) -> Optional[str]:
    for col in df.columns:
        if col.lower() == target.lower():
            return col
    return None


def compute_ohlc_analytics(df: pd.DataFrame, trend_days: int = 30) -> OHLCAnalyticsResult:
    date_col = _find_case_insensitive_column(df, "date")
    close_col = _find_case_insensitive_column(df, "close")
    if not date_col or not close_col:
        raise ValueError("OHLC analytics require Date and Close columns.")

    work = df.copy().sort_values(by=date_col).reset_index(drop=True)
    work["daily_return"] = work[close_col].pct_change()
    work["rolling_mean_7"] = work[close_col].rolling(window=7).mean()
    work["rolling_mean_30"] = work[close_col].rolling(window=30).mean()
    work["rolling_vol_7"] = work["daily_return"].rolling(window=7).std()
    work["rolling_vol_30"] = work["daily_return"].rolling(window=30).std()

    cumulative_max = work[close_col].cummax()
    drawdown = (work[close_col] / cumulative_max) - 1
    work["drawdown"] = drawdown

    min_drawdown_idx = int(drawdown.idxmin()) if len(drawdown.dropna()) else 0
    peak_idx = int(work[close_col].iloc[: min_drawdown_idx + 1].idxmax()) if len(work) else 0

    max_drawdown = float(drawdown.min()) if len(drawdown.dropna()) else 0.0

    daily_ret = work["daily_return"].dropna()
    best_idx = int(daily_ret.idxmax()) if not daily_ret.empty else None
    worst_idx = int(daily_ret.idxmin()) if not daily_ret.empty else None

    last_n = min(trend_days, len(work))
    trend_pct_change = 0.0
    if last_n >= 2:
        start_val = float(work[close_col].iloc[-last_n])
        end_val = float(work[close_col].iloc[-1])
        if start_val != 0:
            trend_pct_change = (end_val / start_val) - 1

    slope = 0.0
    if last_n >= 2:
        y = work[close_col].iloc[-last_n:].astype(float).to_numpy()
        x = np.arange(last_n)
        slope = float(np.polyfit(x, y, 1)[0])

    metrics: Dict[str, object] = {
        "latest_close": float(work[close_col].iloc[-1]) if len(work) else None,
        "latest_date": work[date_col].iloc[-1].strftime("%Y-%m-%d") if len(work) else None,
        "vol_7": float(work["rolling_vol_7"].dropna().iloc[-1]) if len(work["rolling_vol_7"].dropna()) else None,
        "vol_30": float(work["rolling_vol_30"].dropna().iloc[-1]) if len(work["rolling_vol_30"].dropna()) else None,
        "mean_7": float(work["rolling_mean_7"].dropna().iloc[-1]) if len(work["rolling_mean_7"].dropna()) else None,
        "mean_30": float(work["rolling_mean_30"].dropna().iloc[-1]) if len(work["rolling_mean_30"].dropna()) else None,
        "max_drawdown": max_drawdown,
        "max_drawdown_peak_date": work[date_col].iloc[peak_idx].strftime("%Y-%m-%d") if len(work) else None,
        "max_drawdown_trough_date": work[date_col].iloc[min_drawdown_idx].strftime("%Y-%m-%d") if len(work) else None,
        "best_day_return": float(daily_ret.max()) if not daily_ret.empty else None,
        "best_day_date": work[date_col].iloc[best_idx].strftime("%Y-%m-%d") if best_idx is not None else None,
        "worst_day_return": float(daily_ret.min()) if not daily_ret.empty else None,
        "worst_day_date": work[date_col].iloc[worst_idx].strftime("%Y-%m-%d") if worst_idx is not None else None,
        "trend_days": trend_days,
        "trend_pct_change": trend_pct_change,
        "trend_slope": slope,
    }

    return OHLCAnalyticsResult(frame=work, metrics=metrics)
