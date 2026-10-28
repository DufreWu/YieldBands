from datetime import date
import numpy as np
import pandas as pd
import pytest

from yieldbands.data import normalize_dividends, normalize_price, DataSourceError
from yieldbands.engine import GridRule, Settings, build_daily, build_weekly, backtest, conditions


def dividends(*events):
    return pd.DataFrame([{"ex_date": pd.Timestamp(day), "per10": d * 10,
                          "bonus10": b * 10, "cash_per_old_share": d,
                          "split_factor": 1 + b, "announcement": pd.NaT}
                         for day, d, b in events])


def prices(*days, close=10):
    return pd.DataFrame([{"date": pd.Timestamp(day), "open": close, "close": close,
                          "adj_close": close} for day in days])


def test_sina_cash_per_ten_and_missing_planned():
    data = pd.DataFrame({
        "公告日期": ["2024-06-01", "2025-01-01", "2025-02-10"],
        "派息(税前)(元)": [5.0, 4.0, 2.0],
        "送股(股)": [0, 0, 0],
        "转增(股)": [0, 0, 0],
        "进度": ["实施", "预案", "实施"],
        "除权除息日": ["2024-06-10", "--", "2025-02-20"],
    })
    frame, warnings = normalize_dividends(data)
    assert len(frame) == 2
    assert frame.cash_per_old_share.tolist() == [0.5, 0.2]
    assert frame.per10.tolist() == [5, 2]


def test_missing_dividends_must_fail():
    with pytest.raises(DataSourceError):
        normalize_dividends(pd.DataFrame())


def test_split_converts_old_ttm_dividend_to_current_share():
    d = dividends(("2024-01-10", 1.0, 0.0), ("2024-02-10", 0.4, 1.0))
    p = prices("2024-01-09", "2024-01-10", "2024-02-09", "2024-02-10", "2024-02-12", close=10)
    daily = build_daily(p, d)
    # On the second ex-date there are twice as many shares.
    # TTM per current share: (1.0 + 0.4) / 2 = 0.70.
    assert daily.ttm_cash_per_share.iloc[0] == 0
    assert daily.ttm_cash_per_share.iloc[1] == 1.0
    assert daily.ttm_cash_per_share.iloc[3] == pytest.approx(0.7)
    assert daily.yield_pct.iloc[3] == pytest.approx(7.0)


def test_dividend_does_not_leak_into_previous_dates():
    d = dividends(("2024-03-01", 1.0, 0.0))
    p = prices("2024-01-01", "2024-02-28", "2024-03-01")
    daily = build_daily(p, d)
    assert daily.yield_pct.tolist() == [0, 0, 10]


def test_weekly_signals_and_lagged_average():
    dates = pd.bdate_range("2020-01-01", periods=350)
    p = pd.DataFrame({"date": dates, "open": 10.0, "close": 10.0,
                      "adj_close": 10 + np.sin(np.arange(len(dates)) * 0.08)})
    d = dividends(("2019-08-01", 1.0, 0.0))
    daily = build_daily(p, d)
    w = build_weekly(daily, Settings(lookback_years=1, boll_weeks=10))
    assert w.yield_mean.notna().sum() > 0
    idx = w.yield_mean.first_valid_index()
    assert w.loc[idx, "yield_mean"] == pytest.approx(w.loc[idx-52:idx-1, "yield_pct"].mean())


def test_next_session_open_and_dividend_entitlement():
    # Weekly close Friday triggers a buy, executed on Monday. Monday ex-date
    # cash dividend is NOT paid to someone who only bought Monday.
    p = pd.DataFrame({"date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-12"]),
                      "open": [10, 10, 8, 8, 8], "close": [10, 10, 8, 8, 8],
                      "adj_close": [10, 10, 8, 8, 8]})
    d = dividends(("2024-01-08", 1.0, 0))
    daily = build_daily(p, d)
    weekly = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-05"), "price": 10,
        "price_lower": 11, "price_upper": 20,
        "yield_pct": 10, "yield_mean": 5,
    }])
    s = Settings(strategy="price_bb_div", initial_cash=10000, commission=0, stamp_tax=0,
                 slippage=0, lot_size=100)
    result = backtest(daily, weekly, d, date(2024, 1, 4), date(2024, 1, 12), s)
    assert len(result.trades) == 1
    assert result.trades.iloc[0].date == pd.Timestamp("2024-01-08")
    assert result.trades.iloc[0].price == 8
    assert result.metrics["dividend_cash"] == 0
    assert result.portfolio.iloc[-1].shares == 1200
    # benchmark bought on Jan 4 and receives Monday's dividend
    received = result.cash_dividends
    assert received[received.account == "benchmark"].cash.iloc[0] == pytest.approx(900)


def test_signal_is_not_executed_on_same_friday():
    p = prices("2024-01-04", "2024-01-05", close=10)
    d = dividends(("2023-07-01", 1.0, 0.0))
    daily = build_daily(p, d)
    weekly = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-05"), "price": 10,
        "price_lower": 11, "price_upper": 20,
        "yield_pct": 10, "yield_mean": 5,
    }])
    result = backtest(daily, weekly, d, date(2024, 1, 4), date(2024, 1, 5), Settings())
    assert result.trades.empty


def test_weekly_price_buy_sell_logic():
    s = pd.Series({"price": 7, "price_lower": 8, "price_upper": 12,
                   "yield_pct": 5.1, "yield_mean": 4.9})
    assert conditions(s, "price_bb_div") == (True, False)
    s.loc["yield_pct"] = 4.0
    assert conditions(s, "price_bb_div") == (False, False)
    s.loc["price"] = 14.0
    assert conditions(s, "price_bb_div") == (False, True)


def test_normalize_prices_unadjusted_and_qfq_separately():
    raw = pd.DataFrame({"日期": ["2024-01-01", "2024-01-02"], "开盘": [11, 12], "收盘": [10, 11]})
    adj = pd.DataFrame({"日期": ["2024-01-01", "2024-01-02"], "收盘": [8, 9]})
    out = normalize_price(raw, adj)
    assert out.close.tolist() == [10, 11]
    assert out.adj_close.tolist() == [8, 9]


def _grid_case(weeks, grid):
    """Flat 10-yuan prices; `weeks` lists (yield %, band) per Friday, band in {"low", "high", "mid"}."""
    days = pd.bdate_range("2024-01-01", periods=5 * (len(weeks) + 1))
    daily = build_daily(prices(*days, close=10), dividends(("2023-01-01", 0.1, 0.0)))
    fridays = [d for d in days if d.weekday() == 4][:len(weeks)]
    bands = {"low": (11, 15, 20), "high": (1, 3, 9), "mid": (5, 10, 15)}  # lower, mid, upper vs price 10
    weekly = pd.DataFrame([{"date": f, "price": 10, "yield_pct": y,
                            **dict(zip(("price_lower", "price_mid", "price_upper"), bands[b]))}
                           for f, (y, b) in zip(fridays, weeks)])
    s = Settings(strategy="grid", initial_cash=10000, commission=0, stamp_tax=0, slippage=0, grid=grid)
    return backtest(daily, weekly, dividends(("2023-01-01", 0.1, 0.0)), days[0].date(), days[-1].date(), s)


def test_grid_buys_each_level_once_and_sell_frees_levels():
    grid = (GridRule("BUY", 5.0, "≤下轨", 50), GridRule("BUY", 5.5, "≤下轨", 50),
            GridRule("SELL", 4.5, "≥上轨", 100))
    weeks = [(5.2, "low"), (5.2, "low"), (5.6, "low"), (5.6, "mid"), (4.4, "high"), (5.2, "low")]
    t = _grid_case(weeks, grid).trades
    assert t.action.tolist() == ["BUY", "BUY", "SELL", "BUY"]
    # a 5,000-yuan tranche buys 400 shares: 500 would exceed it with the 5-yuan minimum commission
    assert t.shares.tolist() == [400, 400, 800, 400]
    assert t.shares_after.tolist() == [400, 800, 0, 400]
    # each fill is the Monday open after the signal Friday
    assert all(t.date - t.signal_date == pd.Timedelta(days=3))


def test_grid_yield_alone_does_not_buy_without_boll_condition():
    grid = (GridRule("BUY", 5.0, "≤下轨", 50), GridRule("SELL", 4.5, "≥上轨", 100))
    assert _grid_case([(6.0, "mid"), (6.0, "high")], grid).trades.empty


def test_grid_partial_sell_releases_latest_level_first():
    grid = (GridRule("BUY", 5.0, "不限", 50), GridRule("BUY", 5.5, "不限", 50),
            GridRule("SELL", 4.8, "不限", 50), GridRule("SELL", 4.0, "不限", 100))
    weeks = [(5.6, "mid"), (4.7, "mid"), (5.6, "mid"), (3.9, "mid")]
    t = _grid_case(weeks, grid).trades
    # both levels fill in one week; 50% sell frees level 2 only, which then refills;
    # at 3.9% both sell rules match and the larger one (100%) wins
    assert t.action.tolist() == ["BUY", "BUY", "SELL", "BUY", "SELL"]
    assert t.shares_after.tolist() == [400, 800, 400, 800, 0]
    assert t.rule.iloc[2] == "卖 股息率≤4.8%"
