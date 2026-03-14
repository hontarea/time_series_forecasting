"""
LSTM Model Pipeline 

Pipeline:
1. Data Loading
2. Feature Engineering
3. Window Configuration
4. Walk-Forward Validation
5. Backtesting
6. Hyperparameter Optimisation (Optuna)
7. Final Run with Optimised Parameters

Usage:
    python scripts/lstm_pipeline.py
    python scripts/lstm_pipeline.py --no-optimize
    python scripts/lstm_pipeline.py --save-plots plots/
    python scripts/lstm_pipeline.py --save-model models/lstm.pt
    python scripts/lstm_pipeline.py --save-results/lstm.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsf.data.dataset import Dataset
from tsf.data.loader import DataLoader
from tsf.data.feature_engineer import FeatureEngineer
from tsf.data.scaler import FeatureScaler
from tsf.data.window import WindowGenerator
from tsf.models.torch_adapter import TorchAdapter
from tsf.execution.walk_forward import WalkForwardEngine
from tsf.execution.backtester import Backtester
from tsf.execution.strategy import SignBasedStrategy
from tsf.optimization.optimizer import Optimizer
from tsf.utils.evaluation import Metric

from models.lstm import LSTMModel


#  Configuration
DATA_PATH = PROJECT_ROOT / "btc_usdt_1h.csv"

LOOKBACK = 336    # 14 days of 1-hour data
HORIZON = 24      # 1 day ahead

FEATURE_CONFIG = [
    # Momentum
    {"name": "RSI", "period": 14},
    {"name": "KAMA", "period": 30},
    # Trend
    {"name": "EMA", "period": 12},
    {"name": "TEMA", "period": 12},
    # Volatility
    {"name": "ATR", "period": 14},
    {"name": "BBANDS", "period": 20},
    # Volume
    {"name": "OBV"},
    # Label — forward cumulative log return: ln(P_{t+24} / P_t)
    {"name": "log_return", "horizon": HORIZON},
]

WINDOW_CONFIG = {
    "train_window": "4380h",
    "test_window": "1d",
    "step": "1d",
    "mode": "sliding",
}

# Torch training defaults
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING = 5

SEARCH_SPACE = {
    "lr": ("float", 1e-5, 1e-1, {"log": True}),
    "epochs": ("int", 10, 100),
}

N_TRIALS = 10
ANNUAL_FACTOR = 365  # one 24-hour window per day


#  Pipeline steps
def load_data() -> Dataset:
    """Load the BTC/USDT 1H dataset."""
    print("1. LOADING DATA")
    dataset = DataLoader.from_csv(str(DATA_PATH), preset="tick")
    print(dataset)
    print(f"   Rows: {len(dataset):,}")
    return dataset


def engineer_features(dataset: Dataset) -> FeatureEngineer:
    """Apply technical indicators and create the label."""
    print("\n2. FEATURE ENGINEERING")
    engineer = FeatureEngineer(FEATURE_CONFIG)
    engineer.apply(dataset)
    print(f"   Features ({len(dataset.feature_cols)}): {dataset.feature_cols}")
    print(f"   Labels:   {dataset.label_cols}")
    print(f"   Rows after cleanup: {len(dataset):,}")
    return engineer


def configure_window(dataset: Dataset) -> WindowGenerator:
    """Set up the walk-forward sliding window."""
    print("\n3. WINDOW CONFIGURATION")
    window = WindowGenerator(dataset=dataset, **WINDOW_CONFIG)
    print(window)
    print(f"   Summary: {window.summary()}")
    return window


def _build_model(dataset: Dataset, lr: float = LR, epochs: int = EPOCHS) -> TorchAdapter:
    """Construct a LSTMModel wrapped in TorchAdapter."""
    n_features = len(dataset.feature_cols)
    module = LSTMModel(
        seq_len=LOOKBACK,
        pred_len=HORIZON,
        enc_in=n_features,
        c_out=1,
    )
    return TorchAdapter(
        module=module,
        lr=lr,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING,
    )


def run_baseline(
    dataset: Dataset, window: WindowGenerator
) -> tuple[pd.DataFrame, dict, WalkForwardEngine]:
    """Run walk-forward with default LSTM model and backtest."""
    print("\n4. BASELINE WALK-FORWARD + BACKTEST")

    model = _build_model(dataset)
    scaler = FeatureScaler(method="standard")
    engine = WalkForwardEngine(
        model=model, window=window, scaler=scaler, reset_model=True,
        lookback=LOOKBACK, horizon=HORIZON,
    )

    predictions = engine.run(verbose=True)
    print(f"\n   Total predictions: {len(predictions)}")

    if predictions.empty:
        print("   WARNING: No predictions produced.")
        return predictions, {}, engine

    backtester = Backtester(
        dataset=dataset, strategy=SignBasedStrategy(), horizon=HORIZON,
    )
    result = backtester.run(predictions["prediction"], annual_factor=ANNUAL_FACTOR)

    print("\n   === Baseline Backtest Metrics ===")
    for name, value in result["metrics"].items():
        print(f"     {name:20s}: {value:.6f}")

    return predictions, result, engine


def optimize(dataset: Dataset, window: WindowGenerator) -> dict:
    """Run Optuna hyperparameter search for lr and epochs."""
    print("\n5. HYPERPARAMETER OPTIMISATION (Optuna)")

    opt_model = _build_model(dataset)
    opt_scaler = FeatureScaler(method="standard")
    opt_engine = WalkForwardEngine(
        model=opt_model, window=window, scaler=opt_scaler, reset_model=True,
        lookback=LOOKBACK, horizon=HORIZON,
    )

    optimizer = Optimizer(
        engine=opt_engine,
        dataset=dataset,
        metric=Metric.SHARPE_RATIO,
        search_space=SEARCH_SPACE,
        strategy=SignBasedStrategy(),
        n_trials=N_TRIALS,
        direction="maximize",
    )
    optimizer.run()

    print(f"\n   Best params:       {optimizer.best_params}")
    print(f"   Best Sharpe Ratio: {optimizer.best_value:.4f}")
    return optimizer.best_params


def run_optimised(
    dataset: Dataset, window: WindowGenerator, best_params: dict
) -> tuple[pd.DataFrame, dict, WalkForwardEngine]:
    """Re-run the pipeline with optimised parameters."""
    print("\n6. FINAL RUN (Optimised LSTM)")

    best_lr = best_params.get("lr", LR)
    best_epochs = best_params.get("epochs", EPOCHS)
    print(f"   lr = {best_lr:.6g},  epochs = {best_epochs}")

    model = _build_model(dataset, lr=best_lr, epochs=best_epochs)
    scaler = FeatureScaler(method="standard")
    engine = WalkForwardEngine(
        model=model, window=window, scaler=scaler, reset_model=True,
        lookback=LOOKBACK, horizon=HORIZON,
    )

    predictions = engine.run(verbose=True)

    if predictions.empty:
        print("   WARNING: No predictions produced.")
        return predictions, {}, engine

    backtester = Backtester(
        dataset=dataset, strategy=SignBasedStrategy(), horizon=HORIZON,
    )
    result = backtester.run(predictions["prediction"], annual_factor=ANNUAL_FACTOR)

    print("\n   === Optimised Backtest Metrics ===")
    for name, value in result["metrics"].items():
        print(f"     {name:20s}: {value:.6f}")

    return predictions, result, engine

def save_results(
    baseline_result: dict,
    optimised_result: dict | None,
    save_path: Path,
) -> None:
    """Write backtest metrics to a txt file."""
    lines = ["LSTM MODEL PIPELINE - RESULTS", ""]

    lines.append("CONFIG")
    lines.append(f"  LOOKBACK    : {LOOKBACK}")
    lines.append(f"  HORIZON     : {HORIZON}")
    lines.append(f"  LR (default): {LR}")
    lines.append(f"  EPOCHS (def): {EPOCHS}")
    lines.append(f"  BATCH_SIZE  : {BATCH_SIZE}")
    lines.append("")

    if baseline_result:
        lines.append("BASELINE BACKTEST METRICS")
        for name, value in baseline_result["metrics"].items():
            lines.append(f"  {name:20s}: {value:.6f}")
        lines.append("")

    if optimised_result:
        lines.append("OPTIMISED BACKTEST METRICS")
        for name, value in optimised_result["metrics"].items():
            lines.append(f"  {name:20s}: {value:.6f}")
        lines.append("")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))
    print(f"\n   Results saved to {save_path}")


def plot_results(
    baseline_result: dict,
    optimised_result: dict | None = None,
    save_dir: Path | None = None,
) -> None:
    """Generate equity-curve and distribution plots."""
    print("\nPLOTS")

    if not baseline_result:
        print("   No baseline results to plot.")
        return

    # Plot 1: Baseline equity curve
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    strategy_cum = (1 + baseline_result["strategy_returns"]).cumprod() - 1
    market_cum = (1 + baseline_result["market_returns"]).cumprod() - 1

    axes[0].plot(strategy_cum.values, label="Strategy (LSTM)", color="steelblue")
    axes[0].plot(
        market_cum.values, label="Market (Buy & Hold)",
        color="gray", linestyle="--", alpha=0.8,
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title("LSTM Model — Cumulative Return: Strategy vs Market")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        baseline_result["strategy_returns"].values,
        bins=50, alpha=0.7, color="steelblue", edgecolor="black",
    )
    axes[1].set_title("Strategy Returns Distribution")
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "lstm_baseline.png", dpi=150)
        print(f"   Saved: {save_dir / 'lstm_baseline.png'}")
    plt.show()

    # Plot 2: Baseline vs Optimised vs Market
    if optimised_result:
        fig, ax = plt.subplots(figsize=(14, 5))

        baseline_cum = (1 + baseline_result["strategy_returns"]).cumprod() - 1
        optimised_cum = (1 + optimised_result["strategy_returns"]).cumprod() - 1
        market_cum = (1 + optimised_result["market_returns"]).cumprod() - 1

        ax.plot(baseline_cum.values, label="Baseline", alpha=0.7)
        ax.plot(optimised_cum.values, label="Optimised (Optuna)", alpha=0.7)
        ax.plot(
            market_cum.values, label="Market (Buy & Hold)",
            color="gray", linestyle="--", alpha=0.7,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("LSTM — Baseline vs Optimised vs Market")
        ax.set_ylabel("Cumulative Return")
        ax.set_xlabel("Window")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            fig.savefig(save_dir / "lstm_comparison.png", dpi=150)
            print(f"   Saved: {save_dir / 'lstm_comparison.png'}")
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSTM model pipeline for BTC/USDT 1H forecasting.",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip the Optuna hyperparameter optimisation step.",
    )
    parser.add_argument(
        "--save-plots", type=str, default=None,
        help="Directory to save plot images (created if needed).",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save the final trained model (e.g. models/lstm.pt).",
    )
    parser.add_argument(
        "--save-results", type=str, default=None,
        help="Path to save backtest results as a txt file (e.g. results/lstm.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_dir = None
    if args.save_plots:
        save_dir = Path(args.save_plots)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1-3. Data + Features + Window
    dataset = load_data()
    engineer_features(dataset)
    window = configure_window(dataset)

    # 4. Baseline
    baseline_preds, baseline_result, baseline_engine = run_baseline(dataset, window)

    # 5-6. Optimise + Final run
    optimised_result = None
    final_engine = baseline_engine
    if not args.no_optimize:
        best_params = optimize(dataset, window)
        _, optimised_result, final_engine = run_optimised(dataset, window, best_params)

    # Save results
    if args.save_results:
        save_results(baseline_result, optimised_result, Path(args.save_results))

    # Save model
    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_engine.save_model(model_path)
        print(f"\n   Model saved to {model_path}")

    # Plots
    plot_results(baseline_result, optimised_result, save_dir)

if __name__ == "__main__":
    main()
