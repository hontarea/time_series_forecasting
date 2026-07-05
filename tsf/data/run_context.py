"""
Shared run context - the single construction point of everything producer
runs must agree on.

It is responsible for the verified data span + the cross-asset data precondition,
`DatasetSplit` construction, per-model engine configs, HPO search spaces
and `L_max`, the feature-set tag, writing `forecasts/config/` exactly once.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tsf.data.dataset import Dataset
from tsf.data.dataset_split import DatasetSplit, SplitGeometry
from tsf.data.feature_engineer import DEFAULT_FEATURE_SET, FeatureEngineer
from tsf.data.loader import DataLoader
from tsf.data.scaler import FeatureScaler

_HOUR = pd.Timedelta(hours=1)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fail(message: str) -> None:
    raise ValueError(f"RunContext: {message}")


def _assert_hourly_grid(ts: pd.Series, label: str) -> None:
    """Hard-fail unless `ts` is a strictly increasing gap-free hourly grid."""
    diffs = ts.diff().iloc[1:]
    bad = diffs[diffs != _HOUR]
    if len(bad):
        pos = ts.index.get_loc(bad.index[0])
        _fail(
            f"{label}: hourly grid broken at {len(bad)} point(s); first hole "
            f"after {ts.iloc[pos - 1]} (next row {ts.iloc[pos]}).  A mid-series "
            f"hole usually means a NaN-producing feature row was dropped "
            f"(e.g. LOG_VOLUME on a zero-volume outage hour) - resolve it in "
            f"the data pipeline before producing forecasts."
        )


def _resolve_device(requested: Optional[str]) -> str:
    """
    Resolve the training device for TorchAdapter construction (spec §5.6).

    If "cuda" is not available then it is a hard failure.
    """
    import torch

    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        _fail(
            f"device={requested!r} requested but torch.cuda.is_available() is False"
        )
    return requested


# Model zoo - per-model working defaults

@dataclass(frozen=True)
class ModelSpec:
    """Per-model configuration owned by the shared context."""
    key: str
    module_path: str            # "package.module:ClassName"
    lookback: int               # L - input steps for the sequence window
    arch: Dict[str, Any]        # constructor kwargs beyond the shared prefix
    search_space: Dict[str, tuple]
    adapter: Dict[str, Any]     # TorchAdapter kwargs
    data_format: str = "torch_loader"   # "torch_loader" | "tabular" - drives
                                        # the producer's explicit lag step


_DEFAULT_SEARCH_SPACE: Dict[str, tuple] = {
    "lr": ("float", 1e-5, 1e-1, {"log": True}),
    "epochs": ("int", 10, 100),
}
_DEFAULT_ADAPTER: Dict[str, Any] = {
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "early_stopping_patience": 5,
}

MODEL_ZOO: Dict[str, ModelSpec] = {
    "linear": ModelSpec(
        key="linear", module_path="models.linear:LinearModel", lookback=336,
        arch={"individual": False},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
    "dlinear": ModelSpec(
        key="dlinear", module_path="models.linear:DLinearModel", lookback=336,
        arch={"individual": False},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
    "nlinear": ModelSpec(
        key="nlinear", module_path="models.linear:NLinearModel", lookback=336,
        arch={"individual": False},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
    "lstm": ModelSpec(
        key="lstm", module_path="models.lstm:LSTMModel", lookback=336,
        arch={"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
    "gru": ModelSpec(
        key="gru", module_path="models.gru:GRUModel", lookback=336,
        arch={"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
    "itransformer": ModelSpec(
        key="itransformer", module_path="models.itransformer:ITransformerModel",
        lookback=336,
        arch={"d_model": 128, "n_heads": 4, "e_layers": 2, "d_ff": 256, "dropout": 0.1},
        search_space=_DEFAULT_SEARCH_SPACE, adapter=_DEFAULT_ADAPTER,
    ),
}


class RunContext:
    """
    Everything runs must agree on, constructed once and imported everywhere.

    Args:
        data_dir      : directory of master per-asset CSVs
        geometry      : SplitGeometry shared by every run
        cut_fraction  : in-sample fraction before snapping (default 1/2)
        n_trials      : Optuna trials per model (N)
        l_max         : lookback ceiling for the calendar's inner-val
                        viability check 
        feature_config: FeatureEngineer config
        models        : model zoo registry (default MODEL_ZOO)
        target_col    : label column name ("log_return")
        val_ratio / scaler_method / reset_model : engine-level config shared
                        by every model
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        geometry: SplitGeometry,
        cut_fraction: float = 0.5,
        n_trials: int = 25,
        l_max: Optional[int] = None,
        feature_config: Optional[List[Dict[str, Any]]] = None,
        models: Optional[Dict[str, ModelSpec]] = None,
        target_col: str = "log_return",
        val_ratio: float = 0.1,
        scaler_method: str = "standard",
        reset_model: bool = True,
        device: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.geometry = geometry
        self.cut_fraction = float(cut_fraction)
        self.n_trials = int(n_trials)
        self.feature_config = [dict(spec) for spec in (
            feature_config
            if feature_config is not None
            else [*DEFAULT_FEATURE_SET, {"name": "log_return", "horizon": 1}]
        )]
        self.models = dict(models) if models is not None else dict(MODEL_ZOO)
        self.target_col = target_col
        self.horizon = geometry.horizon_h
        self.val_ratio = float(val_ratio)
        self.scaler_method = scaler_method
        self.reset_model = bool(reset_model)
        self.device = _resolve_device(device)

        if not self.models:
            _fail("model zoo is empty")

        self.L_max = int(l_max) if l_max is not None else max(
            spec.lookback for spec in self.models.values()
        )
        over_ceiling = {
            key: spec.lookback
            for key, spec in self.models.items()
            if spec.lookback > self.L_max
        }
        if over_ceiling:
            _fail(
                f"model lookback exceeds the calendar ceiling L_max="
                f"{self.L_max}: {over_ceiling}."
            )

        # Hash of the canonical serialized FE config.
        canonical = json.dumps(self.feature_config, sort_keys=True)
        self.feature_tag = hashlib.sha256(canonical.encode()).hexdigest()[:16]

        # Derive/verify the common post-FE span across master CSVs
        if not self.data_dir.is_dir():
            _fail(f"data_dir {self.data_dir} does not exist")
        asset_paths = sorted(
            p for p in self.data_dir.glob("*.csv") if not p.name.startswith("_")
        )
        if not asset_paths:
            _fail(f"no master CSVs found in {self.data_dir}")
        self.assets = [p.stem for p in asset_paths]

        summaries: Dict[str, dict] = {}
        for path in asset_paths:
            dataset = self.load_and_engineer(path)
            time_series = dataset.df[dataset.time_col]
            _assert_hourly_grid(time_series, f"{path.name} after FE")
            summaries[path.stem] = {
                "first_usable_ts": time_series.iloc[0].isoformat(),
                "last_usable_ts": time_series.iloc[-1].isoformat(),
                "n_usable_rows": len(dataset),
                "feature_columns": list(dataset.feature_cols),
                "label_columns": list(dataset.label_cols),
            }

        # Check that all assets share identical usable spans, row counts and feature columns.
        reference_asset = self.assets[0]
        reference = summaries[reference_asset]
        for asset, summary in summaries.items():
            if summary != reference:
                diff = {k: (reference[k], summary[k])
                        for k in reference if summary[k] != reference[k]}
                _fail(
                    f"cross-asset data precondition violated: {asset} differs "
                    f"from {reference_asset} after FE on {diff}."
                )
        if self.target_col not in reference["label_columns"]:
            _fail(
                f"target_col {self.target_col!r} is not a label column after "
                f"FE (labels: {reference['label_columns']})."
            )
        self.feature_columns: List[str] = list(reference["feature_columns"])
        self.n_usable_rows: int = int(reference["n_usable_rows"])
        self.span: Tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp(reference["first_usable_ts"]),
            pd.Timestamp(reference["last_usable_ts"]),
        )
        self._reference_summary = dict(reference)

        self._split = DatasetSplit(self.span, self.cut_fraction, geometry, self.L_max)

    # Data loading / verification
    def load_and_engineer(self, csv_path: str | Path) -> Dataset:
        """Master CSV -> Dataset with the shared feature set + label applied."""
        dataset = DataLoader.from_csv(str(csv_path), preset="tick")
        FeatureEngineer(self.feature_config).apply(dataset)
        return dataset

    def asset_key(self, csv_path: str | Path) -> str:
        return Path(csv_path).stem

    def assert_feature_tag(
        self, dataset: Dataset, store_root: Optional[str | Path] = None
    ) -> None:
        """
        Assert the dataset carries exactly the shared feature set 
        """
        if list(dataset.feature_cols) != self.feature_columns:
            _fail(
                f"feature columns do not match the shared feature set.\n"
                f"  expected: {self.feature_columns}\n"
                f"  got:      {list(dataset.feature_cols)}"
            )
        if self.target_col not in dataset.label_cols:
            _fail(f"label column {self.target_col!r} missing from dataset")
        if store_root is not None:
            path = Path(store_root) / "config" / "feature_set.json"
            if path.exists():
                frozen = json.loads(path.read_text())
                if frozen.get("tag") != self.feature_tag:
                    _fail(
                        f"feature-set tag mismatch: context has "
                        f"{self.feature_tag}, frozen config has "
                        f"{frozen.get('tag')} ({path})."
                    )

    def assert_data_precondition(self, dataset: Dataset, csv_path: str | Path) -> None:
        """
        Per-run data precondition 
        """
        asset = self.asset_key(csv_path)
        if asset not in self.assets:
            _fail(
                f"{csv_path} is not a master CSV in {self.data_dir} "
                f"(assets: {self.assets})."
            )
        time_series = dataset.df[dataset.time_col]
        _assert_hourly_grid(time_series, f"{asset} after FE")
        observed = {
            "first_usable_ts": time_series.iloc[0].isoformat(),
            "last_usable_ts": time_series.iloc[-1].isoformat(),
            "n_usable_rows": len(dataset),
            "feature_columns": list(dataset.feature_cols),
            "label_columns": list(dataset.label_cols),
        }
        expected = self._reference_summary
        if observed != expected:
            diff = {k: (expected[k], observed[k])
                    for k in expected if observed[k] != expected[k]}
            _fail(
                f"loaded dataset deviates from the verified span/columns "
                f"for {asset}: {diff}"
            )

    # Calendar
    def dataset_split(self) -> DatasetSplit:
        """The shared calendar (construction assertions already passed)."""
        return self._split

    # Model factories
    def model_spec(self, model_key: str) -> ModelSpec:
        if model_key not in self.models:
            _fail(f"unknown model {model_key!r}, zoo: {sorted(self.models)}")
        return self.models[model_key]

    def search_space(self, model_key: str) -> Dict[str, tuple]:
        return dict(self.model_spec(model_key).search_space)

    def build_model(self, model_key: str, n_features: int):
        """Fresh adapter-wrapped model instance"""
        spec = self.model_spec(model_key)
        if spec.data_format != "torch_loader":
            raise NotImplementedError(
                f"model {model_key!r} declares data_format={spec.data_format!r}"
            )
        module_name, class_name = spec.module_path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        module = cls(
            seq_len=spec.lookback,
            pred_len=self.horizon,
            enc_in=n_features,
            c_out=1,
            **spec.arch,
        )
        from tsf.models.torch_adapter import TorchAdapter
        return TorchAdapter(module=module, device=self.device, **spec.adapter)

    def build_engine(self, model_key: str, window):
        """
        Fresh WalkForwardEngine (fresh model, fresh scaler) on `window`.
        """
        from tsf.execution.walk_forward import WalkForwardEngine
        spec = self.model_spec(model_key)
        model = self.build_model(model_key, n_features=len(window.dataset.feature_cols))
        return WalkForwardEngine(
            model=model,
            window=window,
            scaler=FeatureScaler(method=self.scaler_method),
            reset_model=self.reset_model,
            lookback=spec.lookback,
            horizon=self.horizon,
            val_ratio=self.val_ratio,
        )

    def engine_config(self, model_key: str) -> dict:
        """Engine-level configuration for meta.json"""
        spec = self.model_spec(model_key)
        return {
            "lookback": spec.lookback,
            "horizon": self.horizon,
            "val_ratio": self.val_ratio,
            "scaler_method": self.scaler_method,
            "reset_model": self.reset_model,
            "adapter_defaults": dict(spec.adapter),
        }

    # config/ writing 
    def config_payloads(self) -> Dict[str, dict]:
        """The two shared-once artifacts, as JSON-ready dicts"""
        return {
            "calendar": self._split.provenance(),
            "feature_set": {
                "tag": self.feature_tag,
                "feature_config": self.feature_config,
                "feature_columns": self.feature_columns,
                "label_col": self.target_col,
            },
        }

    def ensure_config_written(self, store_root: str | Path) -> Path:
        """
        Write `<store_root>/config/{calendar,feature_set}.json` exactly
        once. If a file already exists it must be deep-equal to this
        context's payload - any drift is a hard failure
        """
        config_dir = Path(store_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in self.config_payloads().items():
            path = config_dir / f"{name}.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if existing != payload:
                    _fail(
                        f"{path} already exists and differs from this context"
                    )
            else:
                path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return config_dir

    def __repr__(self) -> str:
        return (
            f"RunContext(assets={self.assets}, span=({self.span[0]} .. "
            f"{self.span[1]}), tag={self.feature_tag}, L_max={self.L_max}, "
            f"n_trials={self.n_trials}, device={self.device})"
        )


def build_production_context(
    data_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> RunContext:
    """
    The production shared context 
    """
    return RunContext(
        data_dir=Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "final_dataset",
        geometry=SplitGeometry(
            train_window_h=8760,   
            test_window_h=24,
            horizon_h=24,
            delta_insample_h=168,  
            delta_oos_h=24,
            mode="sliding",
        ),
        cut_fraction=0.5,
        n_trials=25,               
        l_max=720,                
        device=device,
    )
