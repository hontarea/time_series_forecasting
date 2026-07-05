from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset


# Registry for technical indicator / returns transforms
_REGISTRY: Dict[str, Callable[..., None]] = {} # ... - we do not care about the arguments a func has


def register(name: str):
    """
    Decorator that adds a transform function to the global registry.

    The decorated function must have the signature:

        def fn(dataset: Dataset, **params) -> None

    It should mutate dataset in place via add_features / add_labels.
    It should not call dropna - the engine does that once at the end.
    """
    def decorator(fn: Callable[..., None]):
        _REGISTRY[name.upper()] = fn
        return fn
    return decorator


# Shared helpers

def _log_returns(series: pd.Series) -> pd.Series:
    """One-step log return  ln(P_t / P_{t-1}).  Causal: uses only P_t and P_{t-1}."""
    return np.log(series / series.shift(1))


def _ewm_vol(returns: pd.Series, span: int) -> pd.Series:
    """
    Exponentially weighted standard deviation of returns over ``span``.
    """
    return returns.ewm(span=span, min_periods=span, adjust=False).std()


# Returns
@register("LOG_RETURN_1")
def _log_return_1(ds: Dataset, column: str = "close", **_kw) -> None:
    ds.add_features(_log_returns(ds.df[column]).rename("log_return_1"))

@register("VOL_SCALED_RETURN")
def _vol_scaled_return(ds: Dataset, column: str = "close", span: int = 72, eps: float = 1e-8, **_kw) -> None:
    """
    Return divided by a recent estimate of its own volatility.

    Puts returns from calm and turbulent periods on a comparable scale. The
    volatility is the exponentially weighted std of returns, ``eps`` guards 
    against division by zero in flat periods.
    """
    r = _log_returns(ds.df[column])
    vol = _ewm_vol(r, span)
    ds.add_features((r / (vol + eps)).rename("vol_scaled_return"))

# Volatility
@register("EWM_VOLATILITY")
def _ewm_volatility(ds: Dataset, column: str = "close", span: int = 72, **_kw) -> None:
    """
    Exponentially weighted std of one-step log returns, as a feature in its own
    right - the current level of market turbulence, direction-independent.
    """
    r = _log_returns(ds.df[column])
    ds.add_features(_ewm_vol(r, span).rename("ewm_volatility"))

# Technical indicators
@register("RSI")
def _rsi(ds: Dataset, period: int = 14, **_kw) -> None:
    """Relative Strength Index (Wilder 1978)"""
    import talib as ta
    ds.add_features(ta.RSI(ds.close, timeperiod=period).rename("rsi"))

@register("MACD_NORM")
def _macd_norm(
    ds: Dataset,
    column: str = "close",
    fast: int = 12,
    slow: int = 26,
    norm_period: int = 72,
    eps: float = 1e-8,
    **_kw,
) -> None:
    """
    Volatility-normalized MACD line.

    Raw MACD = (fast EMA - slow EMA) of price is expressed in price units and
    therefore depends on the price level. Dividing by a recent price-scale
    volatility (rolling std of price) removes that dependence and makes the
    values comparable across assets.
    """
    p = ds.df[column]
    macd = p.ewm(span=fast, adjust=False).mean() - p.ewm(span=slow, adjust=False).mean()
    price_std = p.rolling(window=norm_period, min_periods=norm_period).std()
    ds.add_features((macd / (price_std + eps)).rename("macd_norm"))

# Volume
@register("LOG_VOLUME")
def _log_volume(ds: Dataset, column: str = "volume", **_kw) -> None:
    v = ds.df[column]
    ds.add_features(np.log(v.where(v > 0)).rename("log_volume"))


# Labels
@register("LOG_RETURN")
def _log_return(ds: Dataset, column: str = "close", horizon: int = 1, **_kw) -> None:
    """
    Forward cumulative log return:  ln(P_{t+horizon} / P_t)
    """
    series = ds.df[column]
    log_ret = np.log(series.shift(-horizon) / series)
    ds.add_labels(log_ret.rename("log_return"))

@register("SIMPLE_RETURN")
def _simple_return(ds: Dataset, column: str = "close", shift: int = -1, **_kw) -> None:
    """Percentage change, shifted for forward-looking labels."""
    series = ds.df[column]
    ret = series.pct_change().shift(shift)
    ds.add_labels(ret.rename("simple_return"))


# Tabular helper
@register("LAGGED_RETURNS")
def _lagged_returns(ds: Dataset, column: str = "close", lags: int = 336, **_kw) -> None:
    """
    Create lagged log-return features for tabular (sklearn) models.
    """
    log_ret = _log_returns(ds.df[column])
    for lag in range(1, lags + 1):
        ds.add_features(log_ret.shift(lag).rename(f"log_return_lag_{lag}"))


def add_lagged_returns(dataset: Dataset, lags: int, column: str = "close") -> None:
    """
    Explicit lagged-return feature step for TABULAR (sklearn) models 
    """
    fn = _REGISTRY.get("LAGGED_RETURNS")
    if fn is None:
        raise RuntimeError(
            "Feature engineer 'LAGGED_RETURNS' is not registered in _REGISTRY."
        )
    fn(dataset, lags=lags, column=column)
    dataset.dropna(reset_index=False)


# Canonical shared feature set for all the models
DEFAULT_FEATURE_SET: List[Dict[str, Any]] = [
    {"name": "LOG_RETURN_1"},
    {"name": "VOL_SCALED_RETURN", "span": 72},
    {"name": "EWM_VOLATILITY", "span": 72},
    {"name": "RSI", "period": 14},
    {"name": "MACD_NORM", "fast": 12, "slow": 26, "norm_period": 72},
    {"name": "LOG_VOLUME"},
]


class FeatureEngineer:
    """
    Args:
        config : list[dict]
            Each dict must contain a "name" key matching a registered
            transform. All other keys are forwarded as **params.
        dropna : bool
        reset_index : bool

    Example:
        config = [
            *DEFAULT_FEATURE_SET,
            {"name": "log_return", "horizon": 1},   # forward label
        ]
        FeatureEngineer(config).apply(dataset)
    """

    # Expose the registry so users can register custom transforms
    register = staticmethod(register)

    def __init__(
        self,
        config: List[Dict[str, Any]],
        *,
        dropna: bool = True,
        reset_index: bool = True,
    ):
        self._config = config
        self._dropna = dropna
        self._reset_index = reset_index

    def apply(self, dataset: Dataset) -> Dataset:
        """Apply all configured transforms to dataset in place and return it."""
        for spec in self._config:
            spec = dict(spec)  # Shallow copy so we don't mutate the config
            name = spec.pop("name").upper()
            fn = _REGISTRY.get(name)
            if fn is None:
                raise KeyError(
                    f"Unknown transform '{name}'. "
                    f"Registered transforms: {sorted(_REGISTRY.keys())}"
                )
            fn(dataset, **spec)

        # Single cleanup pass at the end
        if self._dropna:
            dataset.dropna(reset_index=self._reset_index)

        return dataset

    @staticmethod
    def available_transforms() -> List[str]:
        """Return sorted list of registered transform names."""
        return sorted(_REGISTRY.keys())
