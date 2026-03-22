"""
DQN + LSTM Pipeline

Pipeline:
1. Data Loading
2. Feature Engineering
3. Window Configuration
4. [Optional] Hyperparameter Optimisation (Optuna, LSTM via rule-based Backtester)
5. DQN Pipeline (Orchestrator + DQNAgent + TradingEnv)
6. Rule-Based Baseline (Backtester + SignBasedStrategy)
7. Metrics comparison + Plots

Usage:
    python scripts/dqn_pipeline.py
    python scripts/dqn_pipeline.py --no-optimize
    python scripts/dqn_pipeline.py --save-plots plots/
    python scripts/dqn_pipeline.py --save-model models/dqn.pt
    python scripts/dqn_pipeline.py --save-results results/dqn.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from tsf.execution.orchestrator import Orchestrator
from tsf.environments.trading import TradingEnv
from tsf.agents.dqn import DQNAgent
from tsf.optimization.optimizer import Optimizer
from tsf.utils.evaluation import Metric

from models.lstm import LSTMModel


# Configuration
DATA_PATH = PROJECT_ROOT / "btc_usdt_1h.csv"

LOOKBACK, HORIZON = 672, 24

FEATURE_CONFIG = [
    {"name": "RSI",    "period": 14},
    {"name": "KAMA",   "period": 30},
    {"name": "EMA",    "period": 12},
    {"name": "TEMA",   "period": 12},
    {"name": "ATR",    "period": 14},
    {"name": "BBANDS", "period": 20},
    {"name": "OBV"},
    {"name": "log_return", "horizon": 1},
]

WINDOW_CONFIG = {
    "train_window": "4368h",
    "test_window": "1d",
    "step": "1d",
    "mode": "sliding",
}

# LSTM defaults
LR           = 1e-3
EPOCHS       = 50
BATCH_SIZE   = 32
EARLY_STOPPING = 5

SEARCH_SPACE = {
    "lr":     ("float", 1e-5, 1e-1, {"log": True}),
    "epochs": ("int",   10,   100),
}
N_TRIALS = 10

# DQN
OBS_DIM           = HORIZON + 5   # 24 forecast + 5 scalars
DQN_LR            = 1e-4
DQN_EPSILON_DECAY = 5000
DQN_BUFFER_SIZE   = 10_000
DQN_BATCH_SIZE    = 32
DQN_N_LEARN_STEPS = 4

ANNUAL_FACTOR = 8760   # per-hour crypto: 24 × 365


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


def _build_lstm(dataset: Dataset, lr: float = LR, epochs: int = EPOCHS) -> TorchAdapter:
    """Construct an LSTMModel wrapped in TorchAdapter."""
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


def _result_scalars(result: dict, dataset: Dataset | None = None) -> dict:
    """Return scalar metrics + final equity values from a result dict."""
    scalars = dict(result.get("metrics", {}))

    if "equity_curve" in result and len(result["equity_curve"]) > 0:
        scalars["strategy_final_equity"] = float(result["equity_curve"].iloc[-1])

    if "market_returns" in result and len(result["market_returns"]) > 0:
        scalars["market_final_equity"] = float(
            (1 + result["market_returns"]).cumprod().iloc[-1]
        )
    elif dataset is not None and "equity_curve" in result and len(result["equity_curve"]) > 0:
        idx = result["equity_curve"].index
        close = dataset.close
        first_idx, last_idx = idx[0], idx[-1]
        if last_idx + 1 < len(close):
            scalars["market_final_equity"] = float(close.iloc[last_idx + 1] / close.iloc[first_idx])
        else:
            scalars["market_final_equity"] = float(close.iloc[last_idx] / close.iloc[first_idx])

    return scalars


def run_dqn_pipeline(
    dataset: Dataset,
    window: WindowGenerator,
    lr: float = LR,
    epochs: int = EPOCHS,
) -> tuple[dict, DQNAgent]:
    """Run the Orchestrator-based DQN pipeline.

    Returns:
        result : metrics dict from Orchestrator.run()
        agent  : trained DQNAgent (for optional model saving)
    """
    print("\n4. DQN PIPELINE (Orchestrator + DQNAgent + TradingEnv)")

    lstm   = _build_lstm(dataset, lr, epochs)
    scaler = FeatureScaler(method="standard")
    engine = WalkForwardEngine(
        model=lstm, window=window, scaler=scaler,
        reset_model=True, lookback=LOOKBACK, horizon=HORIZON,
    )
    agent = DQNAgent(
        state_dim=OBS_DIM,
        n_actions=3,
        lr=DQN_LR,
        epsilon_decay_steps=DQN_EPSILON_DECAY,
        buffer_size=DQN_BUFFER_SIZE,
        batch_size=DQN_BATCH_SIZE,
    )
    env = TradingEnv(horizon=HORIZON, txn_cost=0.002, carry_position=True, reward_fn="raw")
    orc = Orchestrator(
        engine=engine,
        agent=agent,
        env=env,
        dataset=dataset,
        n_learn_steps=DQN_N_LEARN_STEPS,
        verbose=True,
    )

    result = orc.run(annual_factor=ANNUAL_FACTOR)

    print(f"\n   Final epsilon: {agent.epsilon:.4f}")
    print("\n   === DQN Backtest Results/Metrics ===")
    for name, value in _result_scalars(result, dataset).items():
        print(f"     {name:20s}: {value:.6f}")

    return result, agent


def run_rule_based(
    dataset: Dataset,
    window: WindowGenerator,
    lr: float = LR,
    epochs: int = EPOCHS,
) -> tuple[pd.DataFrame, dict]:
    """Run Backtester + SignBasedStrategy as the rule-based baseline.

    Returns:
        predictions : raw walk-forward prediction DataFrame
        result      : backtest metrics dict
    """
    print("\n5. RULE-BASED BASELINE (SignBasedStrategy)")

    lstm   = _build_lstm(dataset, lr, epochs)
    scaler = FeatureScaler(method="standard")
    engine = WalkForwardEngine(
        model=lstm, window=window, scaler=scaler,
        reset_model=True, lookback=LOOKBACK, horizon=HORIZON,
    )

    predictions = engine.run(verbose=True)
    print(f"\n   Total predictions: {len(predictions)}")

    if predictions.empty:
        print("   WARNING: No predictions produced.")
        return predictions, {}

    backtester = Backtester(
        dataset=dataset, strategy=SignBasedStrategy(), horizon=HORIZON,
    )
    result = backtester.run(predictions["prediction"], annual_factor=ANNUAL_FACTOR)

    print("\n   === Rule-Based Backtest Results/Metrics ===")
    for name, value in _result_scalars(result, dataset).items():
        print(f"     {name:20s}: {value:.6f}")

    return predictions, result


def optimize_lstm(dataset: Dataset, window: WindowGenerator) -> dict:
    """Run Optuna hyperparameter search for LSTM lr and epochs."""
    print("\n[OPT] HYPERPARAMETER OPTIMISATION (Optuna)")

    opt_lstm   = _build_lstm(dataset)
    opt_scaler = FeatureScaler(method="standard")
    opt_engine = WalkForwardEngine(
        model=opt_lstm, window=window, scaler=opt_scaler,
        reset_model=True, lookback=LOOKBACK, horizon=HORIZON,
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


def plot_results(
    dqn_result: dict,
    rule_result: dict | None = None,
    save_dir: Path | None = None,
) -> None:
    """Generate equity-curve and position-histogram plots."""
    print("\nPLOTS")

    if not dqn_result:
        print("   No DQN results to plot.")
        return

    # Figure 1: equity curves + DQN position histogram
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    dqn_equity = dqn_result["equity_curve"]
    axes[0].plot(dqn_equity.values, label="DQN Agent", color="steelblue")

    if rule_result and "equity_curve" in rule_result:
        rule_equity = rule_result["equity_curve"]
        axes[0].plot(rule_equity.values, label="Rule-Based (SignBased)", color="darkorange", alpha=0.8)

    if rule_result and "market_returns" in rule_result:
        market_equity = (1 + rule_result["market_returns"]).cumprod()
        axes[0].plot(market_equity.values, label="Market (Buy & Hold)",
                     color="gray", linestyle="--", alpha=0.7)

    axes[0].axhline(1.0, color="black", linewidth=0.5)
    axes[0].set_title("DQN vs Rule-Based vs Market — Equity Curve (capital starts at 1)")
    axes[0].set_ylabel("Capital")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Position histogram
    if "positions" in dqn_result:
        positions = dqn_result["positions"]
        axes[1].hist(positions.values, bins=[-1.5, -0.5, 0.5, 1.5],
                     color="steelblue", edgecolor="black", alpha=0.8, rwidth=0.6)
        axes[1].set_xticks([-1, 0, 1])
        axes[1].set_xticklabels(["Short (-1)", "Flat (0)", "Long (+1)"])
        axes[1].set_title("DQN Position Distribution")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = save_dir / "dqn_equity.png"
        fig.savefig(path, dpi=150)
        print(f"   Saved: {path}")
    plt.show()

    # Figure 2: side-by-side metrics bar chart (if rule_result available)
    if rule_result and rule_result.get("metrics") and dqn_result.get("metrics"):
        common_keys = [k for k in dqn_result["metrics"] if k in rule_result["metrics"]]
        if common_keys:
            dqn_vals  = [dqn_result["metrics"][k]  for k in common_keys]
            rule_vals = [rule_result["metrics"][k] for k in common_keys]

            x = np.arange(len(common_keys))
            width = 0.35
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.bar(x - width / 2, dqn_vals,  width, label="DQN",        color="steelblue", alpha=0.8)
            ax2.bar(x + width / 2, rule_vals, width, label="Rule-Based", color="darkorange", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(common_keys, rotation=30, ha="right")
            ax2.set_title("DQN vs Rule-Based — Metrics Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            if save_dir:
                path2 = save_dir / "dqn_metrics.png"
                fig2.savefig(path2, dpi=150)
                print(f"   Saved: {path2}")
            plt.show()


def save_results(
    dqn_result: dict,
    rule_result: dict | None,
    save_path: Path,
) -> None:
    """Write metrics comparison to a txt file."""
    lines = ["DQN PIPELINE - RESULTS", ""]

    lines.append("CONFIG")
    lines.append(f"  LOOKBACK         : {LOOKBACK}")
    lines.append(f"  HORIZON          : {HORIZON}")
    lines.append(f"  DQN_LR           : {DQN_LR}")
    lines.append(f"  DQN_EPSILON_DECAY: {DQN_EPSILON_DECAY}")
    lines.append(f"  DQN_BUFFER_SIZE  : {DQN_BUFFER_SIZE}")
    lines.append(f"  DQN_BATCH_SIZE   : {DQN_BATCH_SIZE}")
    lines.append(f"  LSTM_LR (default): {LR}")
    lines.append(f"  LSTM_EPOCHS (def): {EPOCHS}")
    lines.append("")

    if dqn_result and dqn_result.get("metrics"):
        lines.append("DQN METRICS")
        for name, value in dqn_result["metrics"].items():
            lines.append(f"  {name:20s}: {value:.6f}")
        if "equity_curve" in dqn_result and len(dqn_result["equity_curve"]) > 0:
            lines.append(f"  {'final_equity':20s}: {dqn_result['equity_curve'].iloc[-1]:.6f}")
        lines.append("")

    if rule_result and rule_result.get("metrics"):
        lines.append("RULE-BASED METRICS")
        for name, value in rule_result["metrics"].items():
            lines.append(f"  {name:20s}: {value:.6f}")
        if "equity_curve" in rule_result and len(rule_result["equity_curve"]) > 0:
            lines.append(f"  {'final_equity':20s}: {rule_result['equity_curve'].iloc[-1]:.6f}")
        lines.append("")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))
    print(f"\n   Results saved to {save_path}")

    # Equity curves
    equity_data = {}
    if dqn_result and "equity_curve" in dqn_result:
        equity_data["dqn_equity"] = dqn_result["equity_curve"]
    if rule_result and "equity_curve" in rule_result:
        equity_data["rule_equity"] = rule_result["equity_curve"]
    if rule_result and "market_returns" in rule_result:
        equity_data["market_equity"] = (1 + rule_result["market_returns"]).cumprod()
    if equity_data:
        equity_df = pd.DataFrame(equity_data)
        equity_path = save_path.with_suffix(".csv")
        equity_df.to_csv(equity_path)
        print(f"   Equity curves saved to {equity_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DQN + LSTM pipeline for BTC/USDT 1H forecasting.",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip the Optuna LSTM hyperparameter optimisation step.",
    )
    parser.add_argument(
        "--save-plots", type=str, default=None,
        help="Directory to save plot images (created if needed).",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save the trained DQN agent (e.g. models/dqn.pt).",
    )
    parser.add_argument(
        "--save-results", type=str, default=None,
        help="Path to save backtest results as a txt file (e.g. results/dqn.txt).",
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

    # [Optional] Optimise LSTM hyperparameters
    best_lr, best_epochs = LR, EPOCHS
    if not args.no_optimize:
        best_params  = optimize_lstm(dataset, window)
        best_lr      = best_params.get("lr",     LR)
        best_epochs  = best_params.get("epochs", EPOCHS)
        print(f"\n   Using lr={best_lr:.6g}, epochs={best_epochs}")

    # 4. DQN pipeline
    dqn_result, agent = run_dqn_pipeline(dataset, window, best_lr, best_epochs)

    # 5. Rule-based baseline
    _, rule_result = run_rule_based(dataset, window, best_lr, best_epochs)

    # Print comparison table for results/metrics
    print("\n" + "=" * 60)
    print("METRICS COMPARISON")
    print("=" * 60)
    dqn_scalars  = _result_scalars(dqn_result, dataset)
    rule_scalars = _result_scalars(rule_result, dataset) if rule_result else {}
    all_keys = list(dict.fromkeys(list(dqn_scalars) + list(rule_scalars)))
    header = f"{'Metric':<22} {'DQN':>14} {'Rule-Based':>14}"
    print(header)
    print("-" * len(header))
    for key in all_keys:
        dqn_val  = dqn_scalars.get(key, float("nan"))
        rule_val = rule_scalars.get(key, float("nan"))
        print(f"  {key:<20} {dqn_val:>14.6f} {rule_val:>14.6f}")

    # Save results
    if args.save_results:
        save_results(dqn_result, rule_result if rule_result else None, Path(args.save_results))

    # Save DQN model
    if args.save_model:
        model_path = Path(args.save_model)
        agent.save(model_path)
        print(f"\n   DQN model saved to {model_path}")

    # Plots
    plot_results(dqn_result, rule_result if rule_result else None, save_dir)


if __name__ == "__main__":
    main()