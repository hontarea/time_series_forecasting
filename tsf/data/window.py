from __future__ import annotations

from enum import Enum
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset


# Window creation mode                                                    
class WindowMode(Enum):
    """
    How the training window moves across the timeline.
    "sliding" - start and end of the window moves by step on each iteration
    "expanding" - start is fixed, and end moves by step on each iteration
    """
    SLIDING = "sliding"
    EXPANDING = "expanding"


# WindowGenerator                                                    
class WindowGenerator:
    """
    Configurable time-based window for walk-forward validation.

    Args:

    dataset : Dataset
        The full dataset (will not be mutated).
    train_window : str | pd.Timedelta
        Size of the training window, e.g. "30d" or pd.Timedelta(days=30).
    test_window : str | pd.Timedelta
        Size of the testing window that follows each training window.
    step : str | pd.Timedelta
        How far the window advances on each iteration.
    mode : WindowMode | str
        "sliding" (default) or "expanding".
    start : str | pd.Timestamp, optional
        Override the start of the first window (defaults to the earliest
        timestamp in the dataset).
    end : str | pd.Timestamp, optional
        Override the end boundary (defaults to the latest timestamp).

    Example:

        wg = WindowGenerator(
            dataset=ds,
            train_window="30d",
            test_window="1d",
            step="1d",
        )
        for train_ds, test_ds in wg.get_splits():
            model.fit(train_ds.get_features(), train_ds.get_labels())
            preds = model.predict(test_ds.get_features())
    """

    def __init__(
        self,
        dataset: Dataset,
        train_window: str | pd.Timedelta,
        test_window: str | pd.Timedelta,
        step: str | pd.Timedelta,
        mode: WindowMode | str = WindowMode.SLIDING,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ):
        self.dataset = dataset
        self.train_window = pd.to_timedelta(train_window)
        self.test_window = pd.to_timedelta(test_window)
        self.step = pd.to_timedelta(step)
        self.mode = WindowMode(mode) if isinstance(mode, str) else mode

        # Resolve time boundaries from the dataset
        time_series = self._time_series()
        self.start = pd.to_datetime(start) if start else time_series.min()
        self.end = pd.to_datetime(end) if end else time_series.max()

    # Walk-forward split iterator                                        
    def get_splits(self) -> Generator[Tuple[Dataset, Dataset], None, None]:
        """
        Yield (train_dataset, test_dataset) pairs, advancing the window
        by self.step on each iteration.

        Stops when the test window would extend beyond self.end.
        """
        train_start = self.start
        initial_start = self.start  # Remembered for expanding mode

        while True:
            if self.mode == WindowMode.EXPANDING:
                # Start is static, window grows each step
                current_train_window = train_start + self.train_window - initial_start
                train_end = initial_start + current_train_window
                train_slice_start = initial_start
            else:
                train_end = train_start + self.train_window
                train_slice_start = train_start

            test_start = train_end
            test_end = test_start + self.test_window

            # Stop condition: test window exceeds data boundary
            if test_end > self.end:
                break

            train_ds = self.dataset.slice_by_time(train_slice_start, train_end)
            test_ds = self.dataset.slice_by_time(test_start, test_end)

            # Skip if either split is empty 
            if train_ds.empty or test_ds.empty:
                train_start += self.step
                continue

            # Return current window to process it and come back where stopped
            yield train_ds, test_ds

            train_start += self.step

    # Diagnostic helper                                                 
    def summary(self) -> dict:
        """Return a dict describing the window configuration."""
        n_splits = sum(1 for _ in self.get_splits())
        return {
            "mode": self.mode.value,
            "train_window": str(self.train_window),
            "test_window": str(self.test_window),
            "step": str(self.step),
            "start": str(self.start),
            "end": str(self.end),
            "n_splits": n_splits,
        }

    # Internal helpers                                                   
    def _time_series(self) -> pd.Series:
        """Return the time column as a tz-naive datetime Series."""
        ts = self.dataset.df[self.dataset.time_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        return ts

    def __repr__(self) -> str:
        return (
            f"WindowGenerator(mode={self.mode.value}, "
            f"train={self.train_window}, test={self.test_window}, "
            f"step={self.step})"
        )