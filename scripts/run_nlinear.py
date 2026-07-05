"""
SL producer — NLinear (LTSF-Linear, last-value normalization) (spec §5.6).

One run = one (asset, model, seed) unit persisted to the forecast store.
This script owns ONLY the three items below; everything else — calendar,
feature set, engine config, search space, invariants — is imported from the
shared context and the producer template (per-script configuration is how
the 4368h-vs-4380h calendar drift happened).

Usage:
    python scripts/run_nlinear.py data/final_dataset/BTCUSDT.csv
    python scripts/run_nlinear.py data/final_dataset/ETHUSDT.csv --store-root forecasts
    python scripts/run_nlinear.py data/final_dataset/BTCUSDT.csv --device cuda  # fail loud if no GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tsf.execution.producer import run_producer

MODEL = "nlinear"  # fixed per script (spec §5.6)
SEED = 0           # fixed per script (single-seed regime for now)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce NLinear forecasts for one asset into the "
                    "forecast store (HPO + dense in-sample pass + OOS pass).",
    )
    parser.add_argument(
        "csv_path",
        help="Master per-asset CSV (launch parameter), e.g. "
             "data/final_dataset/BTCUSDT.csv",
    )
    parser.add_argument(
        "--store-root", default=None,
        help="Forecast store root (default: <project>/forecasts).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Training device: 'cuda', 'cpu', 'cuda:0', … (default: "
             "auto-detect; an explicit 'cuda' that is unavailable fails loud "
             "instead of silently falling back to CPU).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_producer(
        MODEL, SEED, args.csv_path,
        store_root=args.store_root, device=args.device, verbose=args.verbose,
    )
    print(f"complete: {run_dir}")


if __name__ == "__main__":
    main()
