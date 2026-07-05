"""
Producer template - the seven-step SL production sequence.

One run = one (asset, model, seed) unit.  A concrete producer script
(`scripts/run_<model>.py`) owns only MODEL, SEED and the launch-time
asset/dataset path, every step below is imported, none is defined locally.

Sequence: seed -> load/FE -> shared context / calendar / config write -> HPO on
the sparse in-sample schedule -> OOS pass -> finalize.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from tsf.data.dataset import Dataset
from tsf.data.feature_engineer import add_lagged_returns
from tsf.data.forecast_store import ForecastStoreWriter, run_dir_for
from tsf.data.run_context import PROJECT_ROOT, RunContext, build_production_context
from tsf.optimization.optimizer import Optimizer

DEFAULT_STORE_ROOT = PROJECT_ROOT / "forecasts"


def seed_everything(seed: int) -> None:
    """Seed random / numpy / torch from the run seed.

    The Optuna sampler is seeded separately at study construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_volatility(train_ds: Dataset) -> float:
    """
    Std of close-to-close log returns over the fold's untrimmed
    training window.
    """
    close = train_ds.close.to_numpy()
    if len(close) < 2:
        return 0.0
    return float(np.std(np.log(close[1:] / close[:-1])))


def _build_meta(
    ctx: RunContext, model_key: str, seed: int, optimizer: Optimizer,
    best_params: dict,
) -> dict:
    """Per-run meta.json payload, identity echo added by the writer."""
    spec = ctx.model_spec(model_key)
    study = optimizer.study
    return {
        "frozen_hps": best_params,
        "hpo": {
            "search_space": ctx.search_space(model_key),
            "n_trials": ctx.n_trials,
            "sampler": f"TPESampler(seed={seed})",
            "pruner": "optuna default (MedianPruner)",
            "best_trial_number": study.best_trial.number,
            "best_value": optimizer.best_value,
            "trials": [
                {"number": t.number, "state": t.state.name,
                 "value": t.value, "params": t.params}
                for t in study.trials
            ],
        },
        "engine_config": ctx.engine_config(model_key),
        "architecture": {
            "module_path": spec.module_path,
            "lookback": spec.lookback,
            "kwargs": spec.arch,
        },
        "feature_tag": ctx.feature_tag,
        "environment": {"device": ctx.device},
    }


def run_producer(
    model_key: str,
    seed: int,
    csv_path: str | Path,
    *,
    context: Optional[RunContext] = None,
    store_root: Optional[str | Path] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """
    Execute one SL production unit and persist it to the forecast store.

    Returns the run directory.  Any exception flips ``_status.json`` to
    ``failed`` (partial files kept for diagnosis) and re-raises; the
    directory is safe to re-run - same seed, same calendar, overwrites only
    its own outputs.
    """
    csv_path = Path(csv_path)
    store_root = Path(store_root) if store_root is not None else DEFAULT_STORE_ROOT

    # 1. Seed everything
    seed_everything(seed)

    # 2. Load + FE (+ explicit lagged step for tabular models) + asserts.
    #    The LOG_RETURN label is built with horizon=1 inside the shared
    #    feature config (the per-step forward-return column; the H-step
    #    structure comes from windowing, never from a cumulative label).
    ctx = (
        context if context is not None
        else build_production_context(device=device)
    )
    print(
        f"[device] training on {ctx.device!r} "
        f"(cuda.is_available={torch.cuda.is_available()}, "
        f"cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0})"
    )
    spec = ctx.model_spec(model_key)
    dataset = ctx.load_and_engineer(csv_path)
    ctx.assert_feature_tag(dataset, store_root=store_root)
    ctx.assert_data_precondition(dataset, csv_path)
    if spec.data_format == "tabular":
        add_lagged_returns(dataset, lags=spec.lookback)

    # 3. Shared calendar (construction assertions fire in the context) and
    #    the write-once shared config.
    split = ctx.dataset_split()
    ctx.ensure_config_written(store_root)

    asset = ctx.asset_key(csv_path)
    writer = ForecastStoreWriter(
        store_root,
        asset=asset, model=model_key, seed=seed,
        dataset=dataset, target_col=ctx.target_col, horizon=ctx.horizon,
        boundaries={
            "in_sample": split.boundaries("in_sample_dense"),
            "oos": split.boundaries("oos"),
        },
        t_cut=split.t_cut, t_last=split.t_last,
    )
    writer.begin()

    try:
        # 4. HPO on the sparse in-sample schedule; this engine is never reused.
        import optuna
        hpo_engine = ctx.build_engine(model_key, split.in_sample(dataset))
        optimizer = Optimizer(
            engine=hpo_engine,
            search_space=ctx.search_space(model_key),
            n_trials=ctx.n_trials,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        optimizer.run()
        # The last trial's parameters are generally not the best trial's -
        # the winner is extracted and installed explicitly below.
        best_params = dict(optimizer.best_params)
        if verbose:
            print(f"HPO winner: {best_params} (val loss {optimizer.best_value:.6g})")

        # 5. In-sample dense pass - fresh engine, frozen HPs, per-fold reset.
        dense_engine = ctx.build_engine(model_key, split.in_sample_dense(dataset))
        dense_engine.set_model_params(best_params)
        for fold_result in dense_engine.iter_folds(verbose=verbose):
            writer.write_fold("in_sample", fold_result,
                              fold_volatility(fold_result.train_ds))

        # 6. OOS pass - fresh engine, same frozen HPs.
        oos_engine = ctx.build_engine(model_key, split.oos(dataset))
        oos_engine.set_model_params(best_params)
        for fold_result in oos_engine.iter_folds(verbose=verbose):
            writer.write_fold("oos", fold_result,
                              fold_volatility(fold_result.train_ds))

        # 7. Finalize: write-time invariants 1-7, meta.json, status complete.
        writer.finalize(_build_meta(ctx, model_key, seed, optimizer, best_params))
    except BaseException as exc:
        writer.fail(exc)
        raise

    return run_dir_for(store_root, asset, model_key, seed)
