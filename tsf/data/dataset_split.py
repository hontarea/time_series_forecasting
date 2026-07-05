from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from tsf.data.dataset import Dataset
from tsf.data.window import WindowGenerator, WindowMode

_HOUR = pd.Timedelta(hours=1)


def _require(condition: bool, message: str) -> None:
    """Raise ValueError with a DatasetSplit-prefixed message when violated."""
    if not condition:
        raise ValueError(f"DatasetSplit: {message}")


@dataclass(frozen=True)
class SplitGeometry:
    """
    Walk-forward geometry shared by every run 

    Args:
        train_window_h   : outer training-window length W
        test_window_h    : test-window length T (must equal horizon_h)
        horizon_h        : forecast horizon H — also the embargo length
                           and the production stride
        delta_insample_h : sparse HPO stride δi (a positive multiple of H)
        delta_oos_h      : production stride δ0 (must equal H, so test
                           windows tile without gap or overlap)
        mode             : "sliding" or "expanding" (WindowGenerator mode)
    """
    train_window_h: int
    test_window_h: int
    horizon_h: int
    delta_insample_h: int
    delta_oos_h: int
    mode: str = "sliding"


class DatasetSplit:
    """
    Pure calendar object: the in-sample/OOS cut and the three walk-forward
    schedules every producer run must share 

    Geometry:
      - The cut is snapped so the OOS side holds an integer number of H-hour
        test windows (shift < H, earlier in time, i.e. toward more OOS folds).
      - OOS test windows tile [t_cut, t_last] exactly, first window at the
        cut; OOS training windows extend backwards into the in-sample half.
      - In-sample folds (sparse HPO schedule and dense schedule) are confined
        entirely to [t_first, t_cut] including their test windows, anchored
        at the cut and striding backwards, so any unused remainder falls at
        the stale start of the data.

    Args:
        span         : (t_first, t_last) — first/last usable timestamps of
                       the verified common data span (inclusive, hourly grid)
        cut_fraction : fraction of rows on the in-sample side before snapping
        geometry     : SplitGeometry
        L_max        : largest lookback across the model zoo (scalar) — used
                       by the inner-val viability assertion so the geometry
                       passes or fails for the whole zoo at once
    """

    SCHEDULES = ("in_sample", "in_sample_dense", "oos")

    def __init__(
        self,
        span: Tuple[pd.Timestamp, pd.Timestamp],
        cut_fraction: float,
        geometry: SplitGeometry,
        L_max: int,
    ):
        self.geometry = geometry
        self.cut_fraction = float(cut_fraction)
        self.L_max = int(L_max)

        g = geometry
        W, T, H = g.train_window_h, g.test_window_h, g.horizon_h
        d_in, d_oos = g.delta_insample_h, g.delta_oos_h

        # Geometry sanity
        _require(W > 0 and T > 0 and H > 0, "window lengths must be positive")
        _require(
            T == H,
            f"test_window_h ({T}) must equal horizon_h ({H})"
        )
        _require(
            d_oos == H,
            f"delta_oos_h ({d_oos}) must equal horizon_h ({H})",
        )
        _require(
            d_in >= H and d_in % H == 0,
            f"delta_insample_h ({d_in}) must be a positive multiple of "
            f"horizon_h ({H})",
        )
        try:
            WindowMode(g.mode)
        except ValueError:
            _require(False, f"unknown mode {g.mode!r}; use 'sliding' or 'expanding'.")
        _require(0.0 < self.cut_fraction < 1.0, "cut_fraction must lie in (0, 1)")
        _require(self.L_max > 0, "L_max must be positive")

        # Inner-val viability 
        _require(
            W - H >= 2 * (self.L_max + H),
            f"inner-val viability violated: train_window_h - horizon_h = "
            f"{W - H} < 2*(L_max + H) = {2 * (self.L_max + H)}.  Increase the "
            f"train window or reduce the zoo's largest lookback.",
        )

        # Span on the hourly grid
        t_first, t_last = pd.Timestamp(span[0]), pd.Timestamp(span[1])
        _require(t_first < t_last, "span start must precede span end")
        span_hours = (t_last - t_first) / _HOUR
        _require(
            float(span_hours).is_integer(),
            f"span [{t_first} .. {t_last}] is not a whole number of hours; "
            f"the calendar assumes a gap-free hourly grid.",
        )
        self.t_first, self.t_last = t_first, t_last
        self.n_total_rows = int(span_hours) + 1

        # Shift the raw cut earlier by s < H hours so the OOS
        # side holds an integer number of H-hour test windows.
        n_insample_raw = math.floor(self.cut_fraction * self.n_total_rows)
        n_oos_raw = self.n_total_rows - n_insample_raw
        _require(n_oos_raw >= 1, "cut leaves no rows on the OOS side")
        self.snap_adjustment_h = (H - n_oos_raw % H) % H
        self.t_cut = t_first + (n_insample_raw - self.snap_adjustment_h) * _HOUR
        self.n_oos_rows = n_oos_raw + self.snap_adjustment_h
        _require(
            self.n_oos_rows % H == 0 and self.snap_adjustment_h < H,
            "internal error: cut snapping failed",  # unreachable by construction
        )
        self.n_oos_folds = self.n_oos_rows // H

        n_insample_rows = self.n_total_rows - self.n_oos_rows
        self.n_insample_rows = n_insample_rows
        D = n_insample_rows  

        _require(
            self.n_oos_folds >= 1,
            f"OOS side holds no complete {H}-hour test window "
            f"(rows after snap: {self.n_oos_rows}).",
        )
        # First OOS training window reaches W hours back from the cut, it may
        # extend into the in-sample half but never before the data span.
        _require(
            D >= W,
            f"first OOS training window underflows the data span: only {D} "
            f"in-sample hours before the cut, train_window_h = {W}.",
        )

        # Dense in-sample schedule (δ = H), anchored at the cut
        self.n_dense_folds = (D - W) // H
        _require(
            self.n_dense_folds >= 1,
            f"in-sample dense schedule holds no fold: hours before cut ({D}) "
            f"- train_window_h ({W}) = {D - W} < horizon_h ({H}).",
        )

        # Sparse HPO schedule (δ = δi), anchored at the cut
        _require(
            D >= W + H,
            f"in-sample sparse schedule holds no fold: needs train_window_h "
            f"+ horizon_h = {W + H} hours before the cut, have {D}.",
        )
        self.n_sparse_folds = (D - W - H) // d_in + 1

        # Schedule table
        self._schedules: Dict[str, Tuple[pd.Timestamp, int, int, pd.Timestamp]] = {
            "oos": (
                self.t_cut - W * _HOUR,
                d_oos,
                self.n_oos_folds,
                self.t_last + _HOUR,
            ),
            "in_sample_dense": (
                self.t_cut - (W + self.n_dense_folds * H) * _HOUR,
                H,
                self.n_dense_folds,
                self.t_cut,
            ),
            "in_sample": (
                self.t_cut - (W + H + (self.n_sparse_folds - 1) * d_in) * _HOUR,
                d_in,
                self.n_sparse_folds,
                self.t_cut,
            ),
        }

        # Cross-checks over the derived tables 
        for name in ("in_sample", "in_sample_dense"):
            b = self.boundaries(name)
            _require(
                b["train_start"].min() >= t_first,
                f"{name} schedule underflows the data span "
                f"(earliest train_start {b['train_start'].min()} < {t_first}).",
            )
            _require(
                b["test_end"].max() == self.t_cut,
                f"internal error: {name} schedule is not anchored at the cut",
            )
        oos_b = self.boundaries("oos")
        _require(
            oos_b["test_start"].iloc[0] == self.t_cut
            and oos_b["test_end"].iloc[-1] == self.t_last + _HOUR
            and (oos_b["test_start"].values == oos_b["test_end"].shift(1).values)[1:].all(),
            "internal error: OOS test windows do not tile [t_cut, t_last]",
        )
        _require(
            oos_b["train_start"].min() >= t_first,
            f"OOS schedule underflows the data span "
            f"(earliest train_start {oos_b['train_start'].min()} < {t_first}).",
        )

    # Schedule accessors — bind a Dataset into unmodified WindowGenerators
    def in_sample(self, dataset: Dataset) -> WindowGenerator:
        """Sparse HPO schedule (δ = δi), confined to [t_first, t_cut]"""
        return self._window(dataset, "in_sample")

    def in_sample_dense(self, dataset: Dataset) -> WindowGenerator:
        """Dense in-sample schedule (δ = H), tiles the coverable span up to t_cut"""
        return self._window(dataset, "in_sample_dense")

    def oos(self, dataset: Dataset) -> WindowGenerator:
        """OOS schedule (δ0 = H): test windows tile [t_cut, t_last] exactly"""
        return self._window(dataset, "oos")

    def _window(self, dataset: Dataset, schedule: str) -> WindowGenerator:
        start, step_h, _, end = self._schedules[schedule]
        g = self.geometry
        return WindowGenerator(
            dataset=dataset,
            train_window=f"{g.train_window_h}h",
            test_window=f"{g.test_window_h}h",
            step=f"{step_h}h",
            mode=g.mode,
            start=start,
            end=end,
        )

    # Scheduled fold boundaries
    def boundaries(self, schedule: str) -> pd.DataFrame:
        """
        Scheduled per-fold boundary table for `schedule`
        """
        _require(
            schedule in self.SCHEDULES,
            f"unknown schedule {schedule!r}; choose from {self.SCHEDULES}",
        )
        start, step_h, n_folds, _ = self._schedules[schedule]
        g = self.geometry
        expanding = WindowMode(g.mode) == WindowMode.EXPANDING

        rows = []
        for j in range(n_folds):
            train_end = start + (j * step_h + g.train_window_h) * _HOUR
            rows.append({
                "fold_idx": j,
                "train_start": start if expanding else start + j * step_h * _HOUR,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": train_end + g.test_window_h * _HOUR,
            })
        return pd.DataFrame(rows, columns=[
            "fold_idx", "train_start", "train_end", "test_start", "test_end",
        ])

    # Provenance
    def provenance(self) -> dict:
        """JSON-ready dump of the calendar for `config/calendar.json`"""
        g = self.geometry
        return {
            "t_first": self.t_first.isoformat(),
            "t_last": self.t_last.isoformat(),
            "n_total_rows": self.n_total_rows,
            "cut_fraction": self.cut_fraction,
            "t_cut": self.t_cut.isoformat(),
            "snap_adjustment_h": self.snap_adjustment_h,
            "geometry": {
                "train_window_h": g.train_window_h,
                "test_window_h": g.test_window_h,
                "embargo_h": g.horizon_h,
                "delta_insample_h": g.delta_insample_h,
                "delta_oos_h": g.delta_oos_h,
                "mode": g.mode,
            },
            "L_max": self.L_max,
            "n_folds": {
                "in_sample": self.n_sparse_folds,
                "in_sample_dense": self.n_dense_folds,
                "oos": self.n_oos_folds,
            },
            "n_rows": {
                "in_sample_dense": self.n_dense_folds * g.horizon_h,
                "oos": self.n_oos_rows,
            },
        }

    def __repr__(self) -> str:
        return (
            f"DatasetSplit(t_cut={self.t_cut}, folds="
            f"{{sparse: {self.n_sparse_folds}, dense: {self.n_dense_folds}, "
            f"oos: {self.n_oos_folds}}}, mode={self.geometry.mode})"
        )
