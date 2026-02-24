from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tsf.data.dataset import Dataset


#  Default column presets for common data types                       
_TICK_PRESET: Dict[str, object] = {
    "feature_cols": [],
    "time_col": "open_time_iso",
    "ohlcv_cols": {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    "use_cols": None,  # None ⇒ keep all CSV columns
}

# Note: add more presets in case of different type of data

PRESETS: Dict[str, Dict[str, object]] = {
    "tick": _TICK_PRESET,
}

class DataLoader:
    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        preset: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        time_col: str = "open_time_iso",
        ohlcv_cols: Optional[dict] = None,
        use_cols: Optional[List[str]] = None,
        index_col: Optional[str | int] = None,
        **read_csv_kwargs,
    ) -> Dataset:
        """
        Load a CSV and return a class Dataset.

        Args:
        
        path : str | Path
            Path to the CSV file.
        preset : str, optional
            One of the registered presets (e.g. "tick").
            When provided, its defaults are used for any parameter not
            explicitly supplied by the caller.
        feature_cols, label_cols, time_col, ohlcv_cols :
            Override the preset or provide values directly.
        use_cols : list[str], optional
            If given, only these columns (plus time_col) are kept from
            the raw CSV.  Useful for dropping irrelevant columns early.
        index_col : str | int, optional
            Passed through to pd.read_csv.
        **read_csv_kwargs :
            Any extra keyword arguments forwarded to pd.read_csv.
        """
        # Resolve preset defaults
        defaults = PRESETS.get(preset, {}) if preset else {}

        feature_cols = feature_cols if feature_cols is not None else defaults.get("feature_cols", [])
        label_cols = label_cols if label_cols is not None else []
        time_col = time_col if time_col != "open_time_iso" else defaults.get("time_col", time_col)
        ohlcv_cols = ohlcv_cols if ohlcv_cols is not None else defaults.get("ohlcv_cols")
        use_cols = use_cols if use_cols is not None else defaults.get("use_cols")

        df = pd.read_csv(path, index_col=index_col, **read_csv_kwargs)

        # Optionally filter out the columns
        if use_cols is not None:
            keep = list(dict.fromkeys([time_col] + use_cols))   # Use dict.fromheys to deduplicate, preserve order
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

        # Build and return Dataset
        return Dataset(
            dataframe=df,
            feature_cols=feature_cols,
            label_cols=label_cols,
            time_col=time_col,
            ohlcv_cols=ohlcv_cols,
        )
