from __future__ import annotations

import numpy as np


def transaction_cost_logret(txn_cost: float, position_delta):
    """
    Transaction cost as an additive log-return term.

    Args:
        txn_cost       : one-way transaction cost (e.g. 0.002 = 0.2%).
        position_delta : change in position. Scalar, NumPy array, or pandas
                         Series.
    """
    return np.log(1.0 - txn_cost * np.abs(position_delta))
