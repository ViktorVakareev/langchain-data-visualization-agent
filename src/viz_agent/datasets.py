"""Find, cache and validate CSV datasets - plain pandas, no LLM involved."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class DatasetError(Exception):
    """Unknown file or column, unreadable CSV, wrong column type, ..."""


class DatasetStore:
    """CSV files in ``data_dir``, loaded once and referenced by file name."""

    def __init__(self, data_dir: Optional[os.PathLike | str] = None, default: Optional[str] = None):
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data")).resolve()
        self.default = default or os.getenv("DEFAULT_DATASET") or None
        self.cache: dict[str, pd.DataFrame] = {}

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.data_dir.glob("*.csv")) if self.data_dir.is_dir() else []

    def resolve(self, name: Optional[str] = None) -> str:
        """File name -> validated base name inside data_dir ('' or None = the default dataset)."""
        name = (name or self.default or "").strip()
        if not name:
            files = self.list_files()
            if len(files) == 1:
                return files[0]
            raise DatasetError(f"Which dataset? Available: {', '.join(files) or 'none'}.")
        base = Path(name.replace("\\", "/")).name  # never leave the data folder
        if not base.lower().endswith(".csv"):
            base += ".csv"
        if not (self.data_dir / base).is_file():
            raise DatasetError(f"Dataset '{name}' not found. Available: {', '.join(self.list_files()) or 'none'}.")
        return base

    def load(self, name: Optional[str] = None) -> pd.DataFrame:
        key = self.resolve(name)
        if key not in self.cache:
            try:
                self.cache[key] = pd.read_csv(self.data_dir / key)
            except Exception as exc:
                raise DatasetError(f"Could not read '{key}': {exc}") from exc
        return self.cache[key]

    def column(self, df: pd.DataFrame, name: str, numeric: bool = False) -> str:
        """Validate a column name (case-insensitive) and optionally that it is numeric."""
        if not name:
            raise DatasetError("A column name is required.")
        matches = [c for c in df.columns if c.lower() == name.strip().lower()]
        if not matches:
            raise DatasetError(f"Column '{name}' not found. Columns: {', '.join(df.columns)}.")
        col = matches[0]
        if numeric and not pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetError(f"Column '{col}' is not numeric.")
        return col

    def summary(self, name: Optional[str] = None) -> dict[str, Any]:
        df = self.load(name)
        return {
            "file_name": self.resolve(name), "rows": int(len(df)),
            "columns": [{"name": c, "dtype": str(df[c].dtype), "n_unique": int(df[c].nunique()),
                         "example": _plain(df[c].iloc[0]) if len(df) else None} for c in df.columns],
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


STORE = DatasetStore()


def configure(data_dir: Optional[os.PathLike | str] = None, default: Optional[str] = None) -> DatasetStore:
    """Point the shared store at a folder and/or default dataset (clears the cache)."""
    if data_dir is not None:
        STORE.data_dir = Path(data_dir).resolve()
    STORE.default = default
    STORE.cache.clear()
    return STORE
