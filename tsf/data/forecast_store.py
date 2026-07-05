"""
Forecast store — the single persistence layer between SL production and
every downstream consumer.

One module owns the schema, the ground-truth anchoring, the writer and 
the reader. Consumers never call `read_csv` on store files.

Layout per experiment run:
    <store_root>/<ASSET>/<MODEL>/seed_<SEED>/
        forecasts_oos.csv
        forecasts_insample.csv
        folds.csv
        meta.json
        _status.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset

_HOUR = pd.Timedelta(hours=1)

# Schema 
SPLITS = ("in_sample", "oos")
SPLIT_FILES = {"in_sample": "forecasts_insample.csv", "oos": "forecasts_oos.csv"}
ROW_COLUMNS = ["split", "fold_idx", "timestamp", "step", "forecast", "actual"]
FOLD_COLUMNS = [
    "fold_idx", "split", "train_start", "train_end", "last_train_label_ts",
    "test_start", "test_end", "volatility", "val_loss", "val_active",
]
FOLDS_FILE = "folds.csv"
META_FILE = "meta.json"
STATUS_FILE = "_status.json"
FLOAT_FORMAT = "%.17g"                 
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"       
_FOLD_TS_COLUMNS = ["train_start", "train_end", "last_train_label_ts",
                    "test_start", "test_end"]


class ForecastStoreError(ValueError):
    """Raised when store contents violate the contract"""


def run_dir_for(store_root: str | Path, asset: str, model: str, seed: int) -> Path:
    return Path(store_root) / asset / model / f"seed_{seed}"


def read_status(run_dir: str | Path) -> dict:
    path = Path(run_dir) / STATUS_FILE
    if not path.exists():
        raise ForecastStoreError(f"{path} does not exist")
    return json.loads(path.read_text())


def anchor_fold_predictions(
    dataset: Dataset,
    fold_preds: pd.DataFrame,
    target_col: str,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """
    Anchor one fold's predictions onto the master dataset by integer row
    label: recover the real timestamp from `dataset.time_col` and
    the realized label value from `target_col` at each prediction's row.

    Returns a DataFrame with columns `timestamp`, `predicted`, `actual` 
    in the fold's row order.
    """
    if prediction_col not in fold_preds.columns:
        raise ValueError(
            f"fold_df missing {prediction_col!r} column, got {list(fold_preds.columns)}"
        )
    df = dataset.df
    return pd.DataFrame({
        "timestamp": df.loc[fold_preds.index, dataset.time_col].to_numpy(),
        "predicted": fold_preds[prediction_col].to_numpy(),
        "actual": df.loc[fold_preds.index, target_col].to_numpy(),
    })


# Write-time invariants 

def check_split_column(rows: pd.DataFrame, split: str) -> None:
    """Filename <-> `split` column agreement"""
    values = rows["split"].unique().tolist()
    if values != [split]:
        raise ForecastStoreError(
            f"filename<->split violated: {SPLIT_FILES[split]} "
            f"contains split values {values}, expected only {split!r}."
        )


def check_uniqueness(rows: pd.DataFrame, split: str) -> None:
    """Exactly one row per timestamp within each half"""
    dup = rows["timestamp"][rows["timestamp"].duplicated()]
    if len(dup):
        raise ForecastStoreError(
            f"uniqueness violated in {split}: duplicate "
            f"timestamp(s), first {dup.iloc[0]} "
            f"({len(dup)} duplicate row(s) total)."
        )


def _check_contiguous(ts_sorted: pd.Series, split: str) -> None:
    diffs = ts_sorted.diff().iloc[1:]
    bad = diffs[diffs != _HOUR]
    if len(bad):
        pos = ts_sorted.index.get_loc(bad.index[0])
        raise ForecastStoreError(
            f"exact tiling violated in {split}: coverage gap "
            f"after {ts_sorted.iloc[pos - 1]} (next stored row "
            f"{ts_sorted.iloc[pos]})."
        )


def check_oos_tiling(
    rows: pd.DataFrame, t_cut: pd.Timestamp, t_last: pd.Timestamp, horizon: int
) -> None:
    """OOS must be contiguous, gap-free, spans [t_cut, t_last], n*H rows."""
    ts = rows["timestamp"].sort_values().reset_index(drop=True)
    _check_contiguous(ts, "oos")
    if ts.iloc[0] != t_cut:
        raise ForecastStoreError(
            f"exact tiling violated in oos: first timestamp "
            f"{ts.iloc[0]} != t_cut {t_cut}."
        )
    if ts.iloc[-1] != t_last:
        raise ForecastStoreError(
            f"exact tiling violated in oos: last timestamp "
            f"{ts.iloc[-1]} != t_last {t_last}."
        )
    if len(ts) % horizon != 0:
        raise ForecastStoreError(
            f"exact tiling violated in oos: row count "
            f"{len(ts)} is not a multiple of horizon {horizon}."
        )


def check_insample_coverage(
    rows: pd.DataFrame, t_cut: pd.Timestamp, horizon: int
) -> None:
    """Dense coverage contiguous up to the cut"""
    ts = rows["timestamp"].sort_values().reset_index(drop=True)
    _check_contiguous(ts, "in_sample")
    if ts.iloc[-1] != t_cut - _HOUR:
        raise ForecastStoreError(
            f"exact tiling violated in in_sample: last "
            f"timestamp {ts.iloc[-1]} != t_cut - 1h ({t_cut - _HOUR})."
        )
    if len(ts) % horizon != 0:
        raise ForecastStoreError(
            f"exact tiling violated in in_sample: row count "
            f"{len(ts)} is not a multiple of horizon {horizon}."
        )


def check_half_disjointness(
    insample_rows: pd.DataFrame, oos_rows: pd.DataFrame, t_cut: pd.Timestamp
) -> None:
    """Half disjointness, max(is) < t_cut <= min(oos)."""
    max_is = insample_rows["timestamp"].max()
    min_oos = oos_rows["timestamp"].min()
    if not (max_is < t_cut <= min_oos):
        raise ForecastStoreError(
            f"half disjointness violated: max in-sample "
            f"timestamp {max_is}, t_cut {t_cut}, min OOS timestamp {min_oos}."
        )
    overlap = set(insample_rows["timestamp"]) & set(oos_rows["timestamp"])
    if overlap:
        raise ForecastStoreError(
            f"half disjointness violated: {len(overlap)} "
            f"timestamp(s) present in both halves, e.g. {sorted(overlap)[0]}."
        )


def check_embargo(folds: pd.DataFrame, horizon: int) -> None:
    """Test_start - last_train_label_ts >= embargo (H)."""
    gap = folds["test_start"] - folds["last_train_label_ts"]
    bad = folds[gap < horizon * _HOUR]
    if len(bad):
        row = bad.iloc[0]
        raise ForecastStoreError(
            f"embargo violated: fold "
            f"{int(row['fold_idx'])} ({row['split']}) has "
            f"test_start - last_train_label_ts = "
            f"{row['test_start'] - row['last_train_label_ts']} < {horizon}h."
        )


def check_dual_key(
    rows: pd.DataFrame, folds: pd.DataFrame, split: str, horizon: int
) -> None:
    """
    Dual-key consistency: each row's timestamp lies in its
    fold's [test_start, test_end) and `step` equals its position within the
    fold's H-step block (strong form: timestamp == test_start + step hours).
    """
    fold_bounds = folds[folds["split"] == split].set_index("fold_idx")
    row_folds = set(rows["fold_idx"].unique())
    table_folds = set(fold_bounds.index)
    if row_folds != table_folds:
        raise ForecastStoreError(
            f"dual-key violated in {split}: row fold_idx set "
            f"{sorted(row_folds)} != folds.csv fold_idx set "
            f"{sorted(table_folds)}."
        )
    for fold_idx, group in rows.groupby("fold_idx", sort=True):
        test_start = fold_bounds.loc[fold_idx, "test_start"]
        group = group.sort_values("step")
        if list(group["step"]) != list(range(horizon)):
            raise ForecastStoreError(
                f"dual-key violated in {split} fold "
                f"{fold_idx}: step values {sorted(group['step'])} != "
                f"0..{horizon - 1}."
            )
        expected = np.array(
            [test_start + step * _HOUR for step in range(horizon)],
            dtype="datetime64[ns]",
        )
        actual_ts = group["timestamp"].to_numpy()
        if not (actual_ts == expected).all():
            mismatch = int(np.where(actual_ts != expected)[0][0])
            raise ForecastStoreError(
                f"dual-key violated in {split} fold "
                f"{fold_idx}: step {int(group['step'].iloc[mismatch])} has "
                f"timestamp {group['timestamp'].iloc[mismatch]}, expected "
                f"{expected[mismatch]}."
            )


# Writer

class ForecastStoreWriter:
    """
    Streaming writer for one (asset, model, seed) run.

    Args:
        store_root : store root directory (…/forecasts)
        asset/model/seed : run identity — becomes the run path
        dataset    : the UNSCALED master Dataset (anchoring source)
        target_col : label column holding the per-step forward log-return
        horizon    : H
        boundaries : {"in_sample": df, "oos": df} — SCHEDULED fold boundary
                     tables from the same DatasetSplit that drove the
                     engines (dense schedule for the in-sample half)
        t_cut/t_last : calendar anchors for the tiling invariants
    """

    def __init__(
        self,
        store_root: str | Path,
        *,
        asset: str,
        model: str,
        seed: int,
        dataset: Dataset,
        target_col: str,
        horizon: int,
        boundaries: Dict[str, pd.DataFrame],
        t_cut: pd.Timestamp,
        t_last: pd.Timestamp,
    ):
        self.asset, self.model, self.seed = asset, model, int(seed)
        self.run_dir = run_dir_for(store_root, asset, model, seed)
        self.dataset = dataset
        self.target_col = target_col
        self.horizon = int(horizon)
        self.t_cut, self.t_last = pd.Timestamp(t_cut), pd.Timestamp(t_last)
        missing = [s for s in SPLITS if s not in boundaries]
        if missing:
            raise ForecastStoreError(f"boundaries missing for split(s) {missing}")
        self._bounds = {
            split: df.set_index("fold_idx", drop=False) for split, df in boundaries.items()
        }
        self._fold_rows: List[dict] = []
        self._counts = {s: {"folds": 0, "rows": 0} for s in SPLITS}
        self._started_at: Optional[str] = None
        self._begun = False

    # Lifecycle
    def begin(self) -> "ForecastStoreWriter":
        """Clean this run's own outputs and mark the run as running."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in [*SPLIT_FILES.values(), FOLDS_FILE, META_FILE]:
            (self.run_dir / name).unlink(missing_ok=True)
        self._fold_rows.clear()
        self._counts = {s: {"folds": 0, "rows": 0} for s in SPLITS}
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._begun = True
        self._write_status("running")
        return self

    def write_fold(self, split: str, fold_result, volatility: float) -> None:
        """
        Persist one fold packet.

        `fold_result` is the engine's FoldResult: it must
        carry `fold_preds`, `fold_idx`, `last_train_label_ts`, `val_loss` 
        and `val_active`.
        """
        if not self._begun:
            raise ForecastStoreError("write_fold called before begin()")
        if split not in SPLITS:
            raise ForecastStoreError(f"unknown split {split!r}; use one of {SPLITS}")
        fold_idx = int(fold_result.fold_idx)
        fold_preds = fold_result.fold_preds
        if len(fold_preds) != self.horizon:
            raise ForecastStoreError(
                f"{split} fold {fold_idx}: {len(fold_preds)} prediction rows "
                f"!= horizon {self.horizon} — partial folds are unwritable."
            )
        bounds = self._bounds[split]
        if fold_idx not in bounds.index:
            raise ForecastStoreError(
                f"{split} fold {fold_idx} is not in the scheduled boundary "
                f"table (folds 0..{len(bounds) - 1})."
            )
        anchored = anchor_fold_predictions(self.dataset, fold_preds, self.target_col)

        rows = pd.DataFrame({
            "split": split,
            "fold_idx": fold_idx,
            "timestamp": anchored["timestamp"],
            "step": np.arange(self.horizon, dtype=np.int64),
            "forecast": anchored["predicted"].astype(np.float64),
            "actual": anchored["actual"].astype(np.float64),
        })[ROW_COLUMNS]
        path = self.run_dir / SPLIT_FILES[split]
        rows.to_csv(
            path, mode="a", header=not path.exists(), index=False,
            float_format=FLOAT_FORMAT, date_format=TS_FORMAT,
        )

        b = bounds.loc[fold_idx]
        self._fold_rows.append({
            "fold_idx": fold_idx,
            "split": split,
            "train_start": b["train_start"],
            "train_end": b["train_end"],
            "last_train_label_ts": pd.Timestamp(fold_result.last_train_label_ts),
            "test_start": b["test_start"],
            "test_end": b["test_end"],
            "volatility": float(volatility),
            "val_loss": (
                float(fold_result.val_loss)
                if fold_result.val_loss is not None else np.nan
            ),
            "val_active": bool(fold_result.val_active),
        })
        self._counts[split]["folds"] += 1
        self._counts[split]["rows"] += self.horizon

    def finalize(self, meta: dict) -> None:
        """
        Write `folds.csv` and `meta.json`, re-read every file from disk,
        run write-time invariants against the serialized data, then flip
        `_status.json` to `complete`.
        """
        if not self._begun:
            raise ForecastStoreError("finalize called before begin()")
        for split in SPLITS:
            if self._counts[split]["folds"] == 0:
                raise ForecastStoreError(
                    f"finalize: no folds written for split {split!r} — both "
                    f"halves are required for a complete run."
                )

        folds = pd.DataFrame(self._fold_rows, columns=FOLD_COLUMNS)
        folds.to_csv(
            self.run_dir / FOLDS_FILE, index=False,
            float_format=FLOAT_FORMAT, date_format=TS_FORMAT,
        )

        # Identity echo checked against the run path.
        identity = {"asset": self.asset, "model": self.model, "seed": self.seed}
        for key, expected in identity.items():
            if key in meta and meta[key] != expected:
                raise ForecastStoreError(
                    f"meta identity echo mismatch: meta[{key!r}] = "
                    f"{meta[key]!r} but the run path says {expected!r}."
                )
        meta = {**identity, **meta}
        (self.run_dir / META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str)
        )

        # Validate what actually landed on disk.
        rows_by_split = {
            split: _read_rows_file(self.run_dir / SPLIT_FILES[split], split)
            for split in SPLITS
        }
        folds_disk = _read_folds_file(self.run_dir / FOLDS_FILE)
        self._run_invariants(rows_by_split, folds_disk)

        self._write_status("complete")

    def fail(self, exc: BaseException) -> None:
        """Mark the run failed; keep partial files for diagnosis."""
        try:
            self._write_status("failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass  # never mask the original failure

    # Internals
    def _run_invariants(
        self, rows_by_split: Dict[str, pd.DataFrame], folds: pd.DataFrame
    ) -> None:
        for split in SPLITS:
            check_split_column(rows_by_split[split], split)
            check_uniqueness(rows_by_split[split], split)
        check_oos_tiling(rows_by_split["oos"], self.t_cut, self.t_last,
                         self.horizon)
        check_insample_coverage(rows_by_split["in_sample"], self.t_cut,
                                self.horizon)
        check_half_disjointness(rows_by_split["in_sample"],
                                rows_by_split["oos"], self.t_cut)
        dup = folds.duplicated(subset=["fold_idx", "split"])
        if dup.any():
            raise ForecastStoreError(
                f"folds.csv has duplicate (fold_idx, split) key(s): "
                f"{folds.loc[dup, ['fold_idx', 'split']].values.tolist()}"
            )
        check_embargo(folds, self.horizon)
        for split in SPLITS:
            check_dual_key(rows_by_split[split], folds, split,
                           self.horizon)

    def _write_status(self, status: str, error: Optional[str] = None) -> None:
        finished = status in ("complete", "failed")
        payload = {
            "status": status,
            "started_at": self._started_at,
            "finished_at": datetime.now(timezone.utc).isoformat() if finished else None,
            "error": error,
            "n_folds_oos": self._counts["oos"]["folds"],
            "n_folds_insample": self._counts["in_sample"]["folds"],
            "n_rows_oos": self._counts["oos"]["rows"],
            "n_rows_insample": self._counts["in_sample"]["rows"],
        }
        (self.run_dir / STATUS_FILE).write_text(json.dumps(payload, indent=2))


# Reader — the single entry point for all store access

@dataclass
class StoreUnit:
    """One loaded (asset, model, seed) run."""
    asset: str
    model: str
    seed: int
    rows: Dict[str, pd.DataFrame]   # split -> validated row frame
    folds: pd.DataFrame
    meta: dict
    status: dict
    run_dir: Path


@dataclass
class FoldPacket:
    """One fold, assembled for the runner."""
    fold_idx: int
    forecast: np.ndarray     # (H,)
    actual: np.ndarray       # (H,)
    timestamps: np.ndarray   # (H,) datetime64
    volatility: float


def _read_rows_file(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise ForecastStoreError(f"{path} is missing.")
    rows = pd.read_csv(
        path,
        dtype={"split": str, "fold_idx": "int64", "step": "int64",
               "forecast": "float64", "actual": "float64"},
        parse_dates=["timestamp"],
        float_precision="round_trip",  
    )
    if list(rows.columns) != ROW_COLUMNS:
        raise ForecastStoreError(
            f"schema violated: {path.name} columns "
            f"{list(rows.columns)} != {ROW_COLUMNS}."
        )
    if (not pd.api.types.is_datetime64_any_dtype(rows["timestamp"])
            or rows["timestamp"].isna().any()):
        raise ForecastStoreError(
            f"schema violated: {path.name} has unparseable "
            f"timestamps."
        )
    check_split_column(rows, split)  # split re-check on every load
    return rows


def _read_folds_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ForecastStoreError(f"{path} is missing.")
    folds = pd.read_csv(
        path,
        dtype={"fold_idx": "int64", "split": str, "volatility": "float64",
               "val_loss": "float64", "val_active": str},
        parse_dates=_FOLD_TS_COLUMNS,
        float_precision="round_trip", 
    )
    if list(folds.columns) != FOLD_COLUMNS:
        raise ForecastStoreError(
            f"schema violated: {path.name} columns "
            f"{list(folds.columns)} != {FOLD_COLUMNS}."
        )
    for col in _FOLD_TS_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(folds[col]) or folds[col].isna().any():
            raise ForecastStoreError(
                f"schema violated: {path.name} column "
                f"{col!r} has unparseable timestamps."
            )
    mapping = {"True": True, "False": False}
    bad = ~folds["val_active"].isin(mapping)
    if bad.any():
        raise ForecastStoreError(
            f"schema violated: {path.name} val_active has "
            f"non-boolean value(s) {folds.loc[bad, 'val_active'].unique().tolist()}."
        )
    folds["val_active"] = folds["val_active"].map(mapping).astype(bool)
    return folds


def load_unit(
    store_root: str | Path,
    asset: str,
    model: str,
    seed: int,
    *,
    allow_incomplete: bool = False,
) -> StoreUnit:
    """
    Load one run with the status gate, schema enforcement and the
    filename<->split re-check.

    `allow_incomplete=True` is a debugging override
    """
    run_dir = run_dir_for(store_root, asset, model, seed)
    status = read_status(run_dir)
    if status.get("status") != "complete" and not allow_incomplete:
        raise ForecastStoreError(
            f"status gate: {run_dir} is "
            f"{status.get('status')!r}, not 'complete' — refusing to load "
            f"(pass allow_incomplete=True to debug)."
        )
    # Under the debugging override, files a partial run has not produced yet
    # load as empty typed frames; whatever exists is still schema-checked.
    def _rows_or_empty(split: str) -> pd.DataFrame:
        path = run_dir / SPLIT_FILES[split]
        if not path.exists() and allow_incomplete:
            return pd.DataFrame({
                "split": pd.Series(dtype=str),
                "fold_idx": pd.Series(dtype="int64"),
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "step": pd.Series(dtype="int64"),
                "forecast": pd.Series(dtype="float64"),
                "actual": pd.Series(dtype="float64"),
            })
        return _read_rows_file(path, split)

    rows = {split: _rows_or_empty(split) for split in SPLITS}
    folds_path = run_dir / FOLDS_FILE
    if not folds_path.exists() and allow_incomplete:
        folds = pd.DataFrame(columns=FOLD_COLUMNS)
    else:
        folds = _read_folds_file(folds_path)
    meta_path = run_dir / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta:
        expected = {"asset": asset, "model": model, "seed": int(seed)}
        echo = {k: meta.get(k) for k in expected}
        if echo != expected:
            raise ForecastStoreError(
                f"meta identity echo {echo} does not match the run path "
                f"{expected}."
            )
    return StoreUnit(
        asset=asset, model=model, seed=int(seed),
        rows=rows, folds=folds, meta=meta, status=status, run_dir=run_dir,
    )


def join_features(
    rows: pd.DataFrame,
    dataset: Dataset,
    store_root: str | Path,
) -> pd.DataFrame:
    """
    The only sanctioned path from store rows to engineered features:
    validate the feature-set tag against `config/feature_set.json`, 
    then join by timestamp, every store row must receive features 
    """
    config_path = Path(store_root) / "config" / "feature_set.json"
    if not config_path.exists():
        raise ForecastStoreError(
            f"feature-join guard: {config_path} is missing — "
            f"cannot verify the feature set before joining."
        )
    frozen = json.loads(config_path.read_text())
    if list(dataset.feature_cols) != frozen["feature_columns"]:
        raise ForecastStoreError(
            f"feature-join guard: dataset feature columns "
            f"{list(dataset.feature_cols)} != frozen feature set "
            f"{frozen['feature_columns']}."
        )

    features = dataset.df[[dataset.time_col, *dataset.feature_cols]]
    joined = rows.merge(
        features, left_on="timestamp", right_on=dataset.time_col,
        how="left", validate="many_to_one",
    ).drop(columns=[dataset.time_col] if dataset.time_col != "timestamp" else [])
    missing = joined[dataset.feature_cols[0]].isna().sum() if dataset.feature_cols else 0
    if missing:
        raise ForecastStoreError(
            f"feature join left {missing} store row(s) without features — "
            f"the dataset does not cover the stored timestamps."
        )
    return joined


def load_units(
    store_root: str | Path,
    keys: Sequence[Tuple[str, str, int]],
    *,
    allow_incomplete: bool = False,
) -> Tuple[List[StoreUnit], pd.DataFrame, pd.DataFrame]:
    """
    Multi-unit loading: load every (asset, model, seed) key, assert
    cross-run calendar identity, and return `(units, rows, folds)` with
    run identity re-attached as columns
    """
    if not keys:
        raise ForecastStoreError("load_units: no unit keys given")
    units = [
        load_unit(store_root, a, m, s, allow_incomplete=allow_incomplete)
        for a, m, s in keys
    ]

    boundary_cols = ["fold_idx", "split", "train_start", "train_end",
                     "test_start", "test_end"]

    def boundary_table(unit: StoreUnit) -> pd.DataFrame:
        return (
            unit.folds[boundary_cols]
            .sort_values(["split", "fold_idx"])
            .reset_index(drop=True)
        )

    reference = boundary_table(units[0])
    for unit in units[1:]:
        table = boundary_table(unit)
        if not table.equals(reference):
            raise ForecastStoreError(
                f"cross-run calendar identity violated: "
                f"{unit.asset}/{unit.model}/seed_{unit.seed} has a different "
                f"fold boundary table than "
                f"{units[0].asset}/{units[0].model}/seed_{units[0].seed}."
            )

    def with_identity(frame: pd.DataFrame, unit: StoreUnit) -> pd.DataFrame:
        out = frame.copy()
        out.insert(0, "asset", unit.asset)
        out.insert(1, "model", unit.model)
        out.insert(2, "seed", unit.seed)
        return out

    rows = pd.concat(
        [with_identity(pd.concat([unit.rows[s] for s in SPLITS],
                                 ignore_index=True), unit)
         for unit in units],
        ignore_index=True,
    )
    folds = pd.concat(
        [with_identity(unit.folds, unit) for unit in units], ignore_index=True
    )
    return units, rows, folds


def iter_fold_packets(unit: StoreUnit, split: str) -> Iterator[FoldPacket]:
    """
    The only sanctioned fold feed: yields per-fold packets in strictly
    increasing `fold_idx` — forecast block, actual block, timestamps,
    and the fold's stored ``volatility``.
    """
    if split not in SPLITS:
        raise ForecastStoreError(f"unknown split {split!r}; use one of {SPLITS}")
    rows = unit.rows[split]
    vol = (
        unit.folds[unit.folds["split"] == split]
        .set_index("fold_idx")["volatility"]
    )
    for fold_idx, group in rows.groupby("fold_idx", sort=True):
        group = group.sort_values("step")
        if fold_idx not in vol.index:
            raise ForecastStoreError(
                f"fold {fold_idx} ({split}) has rows but no folds.csv entry."
            )
        yield FoldPacket(
            fold_idx=int(fold_idx),
            forecast=group["forecast"].to_numpy(),
            actual=group["actual"].to_numpy(),
            timestamps=group["timestamp"].to_numpy(),
            volatility=float(vol.loc[fold_idx]),
        )
