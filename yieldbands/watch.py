"""Latest-day snapshot of a stock for the 股息率观察 watchlist.

Every figure reuses the as-of conventions of ``engine.build_daily``: only
implemented ex-dates count, and cash is expressed per *current* share.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

from .engine import Settings, build_weekly

BOLL_FRAMES = (("日", None), ("周", "W-FRI"), ("月", "M"))  # label, period frequency


@dataclass
class Snapshot:
    date: pd.Timestamp
    close: float
    change_pct: float
    ttm_yield: float
    static_yield: float
    percentile: float | None
    percentile_years: float
    boll: dict[str, str | None]
    targets: tuple[tuple[float, float], tuple[float, float]] | None  # (yield %, price)
    signal: str | None


def boll_position(series: pd.Series, n: int = 20, sigma: float = 2.0) -> str | None:
    """Where the latest value sits in its Bollinger band: 上 / 中上 / 中 / 中下 / 下."""
    tail = series.dropna().iloc[-n:]
    if len(tail) < n:
        return None
    mid, std = tail.mean(), tail.std(ddof=0)
    if std == 0:
        return "中"
    b = (tail.iloc[-1] - (mid - sigma * std)) / (2 * sigma * std)
    if b >= 1:
        return "上"
    if b <= 0:
        return "下"
    if b >= 0.6:
        return "中上"
    if b <= 0.4:
        return "中下"
    return "中"


def yield_targets(ttm_cash: float, yield_pct: float, step: float = 0.5):
    """Prices at which the TTM cash would yield the two nearest `step` levels at/above today."""
    if ttm_cash <= 0 or not yield_pct > 0:
        return None
    low = max(step, floor(yield_pct / step) * step)
    return tuple((y, ttm_cash / (y / 100)) for y in (low, low + step))


def static_cash(dividends: pd.DataFrame, year: int) -> float:
    """Cash paid with ex-dates in `year`, per current share (later splits applied)."""
    div = dividends.sort_values("ex_date")
    factors = div.split_factor.to_numpy(dtype=float)
    later = np.cumprod(factors[::-1])[::-1]  # own and every later split factor
    in_year = (div.ex_date.dt.year == year).to_numpy()
    return float((div.cash_per_old_share.to_numpy(dtype=float) / later)[in_year].sum())


def snapshot(daily: pd.DataFrame, dividends: pd.DataFrame, settings: Settings = Settings(),
             years: int = 10) -> Snapshot:
    last = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) > 1 else last
    dividends = dividends[dividends.ex_date <= last.date]

    # TTM is incomplete during the first 365 days of the download, so skip them.
    window = daily[(daily.date >= max(last.date - pd.DateOffset(years=years),
                                      daily.date.iloc[0] + pd.Timedelta(days=365)))]
    hist = window.yield_pct.dropna()
    percentile = float((hist <= last.yield_pct).mean() * 100) if len(hist) >= 250 else None
    covered = (window.date.iloc[-1] - window.date.iloc[0]).days / 365.25 if len(window) else 0.0

    prices = daily.set_index("date").adj_close
    boll = {}
    for label, rule in BOLL_FRAMES:
        series = prices if rule is None else prices.groupby(prices.index.to_period(rule)).last()
        boll[label] = boll_position(series, settings.boll_weeks, settings.boll_sigma)

    weekly = build_weekly(daily, settings).iloc[-1]
    signal = "买入信号" if weekly.buy_setup else "卖出信号" if weekly.sell_setup else None

    return Snapshot(
        date=last.date, close=float(last.close),
        change_pct=float(last.adj_close / prev.adj_close * 100 - 100),
        ttm_yield=float(last.yield_pct),
        static_yield=static_cash(dividends, last.date.year - 1) / float(last.close) * 100,
        percentile=percentile, percentile_years=covered,
        boll=boll, targets=yield_targets(float(last.ttm_cash_per_share), float(last.yield_pct)),
        signal=signal,
    )
