from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class CSVLoadResult:
    dataframe: pd.DataFrame
    dataset_type: str
    notes: Dict[str, str]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def _find_by_aliases(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    normalized_aliases = {_normalize_name(a) for a in aliases}
    for col in df.columns:
        if _normalize_name(col) in normalized_aliases:
            return col
    return None


def _schema_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    date_col = _find_by_aliases(df, ["date", "datetime", "timestamp", "time"])
    close_col = _find_by_aliases(
        df,
        ["close", "adj close", "adjusted close", "close/last", "last", "price"],
    )
    amount_col = _find_by_aliases(df, ["amount", "value", "total", "transaction amount"])
    category_col = _find_by_aliases(df, ["category", "type", "label"])

    return {
        "date": date_col,
        "close": close_col,
        "amount": amount_col,
        "category": category_col,
    }


def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = _schema_columns(df)
    if cols["date"] and cols["close"]:
        return "ohlcv"
    if cols["date"] and cols["amount"] and cols["category"]:
        return "transactions"
    return "generic"


def _find_case_insensitive_column(df: pd.DataFrame, target: str) -> str | None:
    for col in df.columns:
        if col.lower() == target.lower():
            return col
    return None


def validate_and_clean_financial_csv(csv_path: str | Path) -> CSVLoadResult:
    path = Path(csv_path)
    raw = pd.read_csv(path)
    df = _standardize_columns(raw)

    dataset_type = detect_dataset_type(df)
    notes: Dict[str, str] = {}

    if dataset_type == "generic":
        for col in df.columns:
            maybe_num = pd.to_numeric(df[col], errors="coerce")
            non_null = int(maybe_num.notna().sum())
            if non_null >= max(3, len(df) // 10):
                df[col] = maybe_num
        return CSVLoadResult(dataframe=df, dataset_type=dataset_type, notes=notes)

    schema = _schema_columns(df)
    date_col = schema["date"]
    if not date_col:
        raise ValueError("Missing required date column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=[date_col]).copy()
    dropped_dates = before_drop - len(df)
    if dropped_dates:
        notes["dropped_invalid_dates"] = str(dropped_dates)

    if dataset_type == "ohlcv":
        close_col = schema["close"]
        if not close_col:
            raise ValueError("OHLCV dataset must include Close column.")

        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
            date_col = "Date"
        if close_col != "Close":
            df = df.rename(columns={close_col: "Close"})
            close_col = "Close"

        numeric_candidates = [
            _find_case_insensitive_column(df, name)
            for name in ["open", "high", "low", "close", "volume"]
        ]
        numeric_cols = [c for c in numeric_candidates if c]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before_close_drop = len(df)
        df = df.dropna(subset=[close_col]).copy()
        dropped_close = before_close_drop - len(df)
        if dropped_close:
            notes["dropped_invalid_close"] = str(dropped_close)

        df = df.sort_values(by=date_col).reset_index(drop=True)

    if dataset_type == "transactions":
        amount_col = schema["amount"]
        category_col = schema["category"]
        if not amount_col or not category_col:
            raise ValueError("Transactions dataset must include amount and category columns.")

        if date_col != "date":
            df = df.rename(columns={date_col: "date"})
            date_col = "date"
        if amount_col != "amount":
            df = df.rename(columns={amount_col: "amount"})
            amount_col = "amount"
        if category_col != "category":
            df = df.rename(columns={category_col: "category"})
            category_col = "category"

        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        before_amount_drop = len(df)
        df = df.dropna(subset=[amount_col]).copy()
        dropped_amount = before_amount_drop - len(df)
        if dropped_amount:
            notes["dropped_invalid_amount"] = str(dropped_amount)

        df[category_col] = df[category_col].astype(str).str.strip().fillna("Unknown")
        df = df.sort_values(by=date_col).reset_index(drop=True)

    return CSVLoadResult(dataframe=df, dataset_type=dataset_type, notes=notes)
