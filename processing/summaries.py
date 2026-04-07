from __future__ import annotations

from typing import Dict, List

import pandas as pd


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def generate_finance_summary_snippets(metrics: Dict[str, object], file_name: str) -> List[str]:
    snippets: List[str] = []

    snippets.append(
        f"Dataset {file_name}: latest close is {metrics.get('latest_close')} on {metrics.get('latest_date')}."
    )

    snippets.append(
        "7-day rolling mean is "
        f"{metrics.get('mean_7')} and 30-day rolling mean is {metrics.get('mean_30')}."
    )

    snippets.append(
        "7-day volatility is "
        f"{pct(metrics.get('vol_7'))} and 30-day volatility is {pct(metrics.get('vol_30'))}."
    )

    snippets.append(
        "Max drawdown is "
        f"{pct(metrics.get('max_drawdown'))} from {metrics.get('max_drawdown_peak_date')} "
        f"to {metrics.get('max_drawdown_trough_date')}."
    )

    snippets.append(
        "Best day return was "
        f"{pct(metrics.get('best_day_return'))} on {metrics.get('best_day_date')}; "
        "worst day return was "
        f"{pct(metrics.get('worst_day_return'))} on {metrics.get('worst_day_date')}."
    )

    snippets.append(
        f"Trend over last {metrics.get('trend_days')} days: "
        f"percent change {pct(metrics.get('trend_pct_change'))}, "
        f"slope {metrics.get('trend_slope'):.6f}."
    )

    return snippets


def finance_snippets_to_documents(snippets: List[str], file_name: str) -> List[Dict]:
    docs: List[Dict] = []
    for idx, snippet in enumerate(snippets):
        chunk_id = f"csv::{file_name}::summary::{idx}"
        docs.append(
            {
                "text": snippet,
                "metadata": {
                    "source_type": "csv_summary",
                    "file_name": file_name,
                    "page_number": None,
                    "chunk_id": chunk_id,
                },
            }
        )
    return docs


def generate_generic_csv_summary_snippets(df: pd.DataFrame, file_name: str) -> List[str]:
    snippets: List[str] = []
    snippets.append(f"Dataset {file_name} has {len(df)} rows and {len(df.columns)} columns.")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    snippets.append(
        f"Numeric columns detected: {', '.join(numeric_cols[:10]) if numeric_cols else 'none'}."
    )

    for col in numeric_cols[:8]:
        series = df[col].dropna()
        if series.empty:
            continue
        snippets.append(
            f"Column {col}: mean={series.mean():.4f}, median={series.median():.4f}, "
            f"min={series.min():.4f}, max={series.max():.4f}."
        )

    categorical_cols = [
        c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c]))
    ]
    for col in categorical_cols[:5]:
        top_values = df[col].astype(str).value_counts(dropna=False).head(3)
        if top_values.empty:
            continue
        top_text = ", ".join([f"{idx} ({val})" for idx, val in top_values.items()])
        snippets.append(f"Column {col} top values: {top_text}.")

    return snippets
