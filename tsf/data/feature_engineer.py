from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from tsf.data.dataset import Dataset


# Registry for technical indicator / returns                       
_REGISTRY: Dict[str, Callable[..., None]] = {} # ... - we do not care about the arguments a func has


def register(name: str):
    """
    Decorator that adds a transform function to the global registry.

    The decorated function must have the signature:

        def fn(dataset: Dataset, **params) -> None

    It should mutate dataset in place via add_features / add_labels.
    It should not call dropna — the engine does that once at the end.
    """
    def decorator(fn: Callable[..., None]):
        _REGISTRY[name.upper()] = fn
        return fn
    return decorator


# Technical Indicators                                               

# Momentum  

@register("RSI")
def _rsi(ds: Dataset, period: int = 14, **_kw) -> None:
    import talib as ta
    feature = ta.RSI(ds.close, timeperiod=period).rename("rsi")
    ds.add_features(feature)


@register("KAMA")
def _kama(ds: Dataset, period: int = 30, **_kw) -> None:
    import talib as ta
    feature = ta.KAMA(ds.close, timeperiod=period).rename("kama")
    ds.add_features(feature)


@register("SWMA")
def _swma(ds: Dataset, **_kw) -> None:
    c = ds.close
    swma = (c + 2 * c.shift(1) + 2 * c.shift(2) + c.shift(3)) / 6
    ds.add_features(swma.rename("swma"))


@register("HLC3")
def _hlc3(ds: Dataset, **_kw) -> None:
    hlc3 = ((ds.high + ds.low + ds.close) / 3).rename("hlc3")
    ds.add_features(hlc3)


# Trend 

@register("EMA")
def _ema(ds: Dataset, period: int = 12, **_kw) -> None:
    import talib as ta
    feature = ta.EMA(ds.close, timeperiod=period).rename("ema")
    ds.add_features(feature)


@register("TEMA")
def _tema(ds: Dataset, period: int = 12, **_kw) -> None:
    import talib as ta
    feature = ta.TEMA(ds.close, timeperiod=period).rename("tema")
    ds.add_features(feature)


# Volatility  

@register("ATR")
def _atr(ds: Dataset, period: int = 14, **_kw) -> None:
    import talib as ta
    feature = ta.ATR(ds.high, ds.low, ds.close, timeperiod=period).rename("atr")
    ds.add_features(feature)


@register("BBANDS")
def _bbands(ds: Dataset, period: int = 20, nbdevup: int = 2, nbdevdn: int = 2, matype: int = 0, **_kw) -> None:
    import talib as ta
    upper, middle, lower = ta.BBANDS(
        ds.close, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype
    )
    ds.add_features(upper.rename("bb_upper"))
    ds.add_features(middle.rename("bb_middle"))
    ds.add_features(lower.rename("bb_lower"))


# Volume

@register("OBV")
def _obv(ds: Dataset, **_kw) -> None:
    import talib as ta
    feature = ta.OBV(ds.close, ds.volume).rename("obv")
    ds.add_features(feature)


@register("MFI")
def _mfi(ds: Dataset, period: int = 14, **_kw) -> None:
    import talib as ta
    feature = ta.MFI(ds.high, ds.low, ds.close, ds.volume, timeperiod=period).rename("mfi")
    ds.add_features(feature)


# Label transforms 

@register("LOG_RETURN")
def _log_return(ds: Dataset, column: str = "close", shift: int = -1, **_kw) -> None:
    """
    Compute logarithmic return: ln(price_t / price_{t-1}), then shift
    by "shift" to create a forward-looking label or a lagged feature.
    """
    series = ds.df[column]
    log_ret = np.log(series / series.shift(1))
    if shift != 0:
        log_ret = log_ret.shift(shift)
    ds.add_labels(log_ret.rename("log_return"))


@register("SIMPLE_RETURN")
def _simple_return(ds: Dataset, column: str = "close", shift: int = -1, **_kw) -> None:
    """Percentage change, optionally shifted for forward-looking labels."""
    series = ds.df[column]
    ret = series.pct_change()
    if shift != 0:
        ret = ret.shift(shift)
    ds.add_labels(ret.rename("simple_return"))


class FeatureEngineer:
    """
    Args:

    config : list[dict]
        Each dict must contain a "name" key matching a registered
        transform.  All other keys are forwarded as **params.
    dropna : bool
        Whether to drop NaN rows after applying all transforms (default True).
    reset_index : bool
        Whether to reset the index after dropping NaNs (default True).

    Example:
        config = [
            {"name": "RSI", "period": 14},
            {"name": "BBANDS"},
            {"name": "log_return"},
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
