import numpy as np
import pandas as pd
import pytest

from yieldbands.engine import build_daily
from yieldbands.watch import boll_position, snapshot, static_cash, yield_targets


def dividends(*events):
    return pd.DataFrame([{"ex_date": pd.Timestamp(day), "per10": d * 10,
                          "bonus10": b * 10, "cash_per_old_share": d,
                          "split_factor": 1 + b, "announcement": pd.NaT}
                         for day, d, b in events])


def test_boll_position_levels():
    flat = pd.Series([10.0] * 19 + [10.0])
    assert boll_position(flat) == "中"
    base = pd.Series(np.tile([9.0, 11.0], 10))
    assert boll_position(pd.concat([base, pd.Series([20.0])], ignore_index=True)) == "上"
    assert boll_position(pd.concat([base, pd.Series([0.0])], ignore_index=True)) == "下"
    assert boll_position(pd.Series([10.0] * 5)) is None


def test_yield_targets_round_down_to_half_percent():
    (y1, p1), (y2, p2) = yield_targets(3.0, 8.07)
    assert (y1, y2) == (8.0, 8.5)
    assert p1 == pytest.approx(37.5) and p2 == pytest.approx(3.0 / 0.085)
    assert yield_targets(0.0, 0.0) is None


def test_static_cash_applies_later_splits():
    div = dividends(("2024-06-01", 1.0, 0.0), ("2024-07-01", 0.0, 1.0), ("2025-06-01", 0.4, 0.0))
    assert static_cash(div, 2024) == pytest.approx(0.5)  # 1.0 per old share → 0.5 after 10送10
    assert static_cash(div, 2025) == pytest.approx(0.4)


def test_snapshot_on_constant_price():
    days = pd.bdate_range("2014-01-01", "2025-12-31")
    prices = pd.DataFrame({"date": days, "open": 10.0, "close": 10.0, "adj_close": 10.0})
    div = dividends(*[(f"{y}-06-10", 0.5, 0.0) for y in range(2014, 2026)])
    snap = snapshot(build_daily(prices, div), div, years=10)
    assert snap.ttm_yield == pytest.approx(5.0)
    assert snap.static_yield == pytest.approx(5.0)
    assert snap.change_pct == pytest.approx(0.0)
    assert snap.percentile == pytest.approx(100.0)
    assert snap.boll == {"日": "中", "周": "中", "月": "中"}
    assert [y for y, _ in snap.targets] == [5.0, 5.5]
