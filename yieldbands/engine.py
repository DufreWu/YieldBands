"""As-of dividend yield, weekly signals and next-session-open backtests.

This is educational research software, not a brokerage execution system.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import floor

import numpy as np
import pandas as pd

from .data import StockData

STRATEGIES = {
    "price_bb_div": "价格周线下轨买 / 上轨卖 + 高于X年平均股息率过滤",
    "yield_bb": "股息率周线上轨买 / 下轨卖 + 高于X年平均股息率过滤",
    "yield_mean": "股息率上穿X年均线买 / 下穿均线卖",
    "price_bb": "纯价格周线下轨买 / 上轨卖（对照）",
}


@dataclass(frozen=True)
class Settings:
    strategy: str = "price_bb_div"
    lookback_years: int = 3
    boll_weeks: int = 20
    boll_sigma: float = 2.0
    initial_cash: float = 100_000.0
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.0005
    lot_size: int = 100


@dataclass
class Backtest:
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    cash_dividends: pd.DataFrame
    metrics: dict[str, float | int]


def build_daily(prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """TTM cash dividend yield = cash per *current* share paid over last 365d / raw close.

    For share bonuses, convert past dividends into current-share units at each ex-date.
    Only actual implemented ex-dates <= observation date can affect an observation.
    The TTM cash yield uses known, actually executed payouts, not next-year forecasts.
    """
    frame = prices.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    div = dividends.copy().sort_values("ex_date")
    div["ex_date"] = pd.to_datetime(div["ex_date"])
    events = list(div.itertuples(index=False))
    active: list[tuple[pd.Timestamp, float]] = []
    div_pos = 0
    yields, ttm = [], []
    for row in frame.itertuples(index=False):
        day = row.date
        while div_pos < len(events) and events[div_pos].ex_date <= day:
            ev = events[div_pos]
            split_factor = float(ev.split_factor)
            # Share count is multiplied on the ex-date. Earlier cash dividends
            # must be converted to a per-current-share basis.
            if split_factor != 1.0:
                active = [(d, payout / split_factor) for d, payout in active]
            if float(ev.cash_per_old_share) > 0:
                active.append((ev.ex_date, float(ev.cash_per_old_share) / split_factor))
            div_pos += 1
        cutoff = day - timedelta(days=365)
        active = [(d, payout) for d, payout in active if d > cutoff]
        total = sum(payout for _, payout in active)
        ttm.append(total)
        yields.append(total / row.close * 100.0 if row.close > 0 else np.nan)
    frame["ttm_cash_per_share"] = ttm
    frame["yield_pct"] = yields
    return frame


def build_weekly(daily: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if settings.strategy not in STRATEGIES:
        raise ValueError(f"未知策略：{settings.strategy}")
    if settings.boll_weeks < 2 or settings.lookback_years < 1:
        raise ValueError("布林周期至少2周，历史平均周期至少1年。")
    weekly = (
        daily.assign(week_end=daily["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize())
        .groupby("week_end", as_index=False)
        .agg(date=("date", "last"), price=("adj_close", "last"),
             raw_close=("close", "last"), yield_pct=("yield_pct", "last"))
        .sort_values("date").reset_index(drop=True)
    )
    years_weeks = 52 * settings.lookback_years
    # Always compute a *lagged* historical mean, excluding the current week's yield.
    weekly["yield_mean"] = weekly["yield_pct"].shift(1).rolling(years_weeks, min_periods=years_weeks).mean()
    n = settings.boll_weeks
    sigma = settings.boll_sigma
    for name in ("price", "yield_pct"):
        avg = weekly[name].rolling(n, min_periods=n).mean()
        std = weekly[name].rolling(n, min_periods=n).std(ddof=0)
        weekly[f"{name}_mid"] = avg
        weekly[f"{name}_upper"] = avg + sigma * std
        weekly[f"{name}_lower"] = avg - sigma * std
    setups = [conditions(row, settings.strategy) for _, row in weekly.iterrows()]
    weekly["buy_setup"] = [item[0] for item in setups]
    weekly["sell_setup"] = [item[1] for item in setups]
    return weekly


def conditions(row: pd.Series, strategy: str) -> tuple[bool, bool]:
    """Each condition is evaluated at the actual final trading day of a week."""
    if strategy == "price_bb_div":
        need = ["price", "price_lower", "price_upper", "yield_pct", "yield_mean"]
        if row[need].isna().any():
            return False, False
        return bool(row.price <= row.price_lower and row.yield_pct > row.yield_mean), bool(row.price >= row.price_upper)
    if strategy == "price_bb":
        need = ["price", "price_lower", "price_upper"]
        if row[need].isna().any():
            return False, False
        return bool(row.price <= row.price_lower), bool(row.price >= row.price_upper)
    if strategy == "yield_bb":
        need = ["yield_pct", "yield_upper", "yield_lower", "yield_mean"]
        if row[need].isna().any():
            return False, False
        return bool(row.yield_pct >= row.yield_upper and row.yield_pct > row.yield_mean), bool(row.yield_pct <= row.yield_lower)
    if strategy == "yield_mean":
        need = ["yield_pct", "yield_mean"]
        if row[need].isna().any():
            return False, False
        return bool(row.yield_pct > row.yield_mean), bool(row.yield_pct < row.yield_mean)
    raise ValueError(f"未知策略：{strategy}")


def _commission(notional: float, rate: float) -> float:
    return max(5.0, notional * rate) if notional > 0 else 0.0


def _buy(cash: float, open_price: float, settings: Settings) -> tuple[int, float, float, float]:
    px = open_price * (1 + settings.slippage)
    max_lots = floor(cash / (px * settings.lot_size * (1 + settings.commission)))
    while max_lots > 0:
        qty = int(max_lots * settings.lot_size)
        notional = qty * px
        fee = _commission(notional, settings.commission)
        if notional + fee <= cash + 1e-8:
            return qty, cash - notional - fee, fee, px
        max_lots -= 1
    return 0, cash, 0.0, px


def _sell(cash: float, shares: float, open_price: float, settings: Settings) -> tuple[float, float, float]:
    px = open_price * (1 - settings.slippage)
    notional = shares * px
    fee = _commission(notional, settings.commission) + notional * settings.stamp_tax
    return cash + notional - fee, fee, px


def _mdd(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float((series / series.cummax() - 1).min() * 100)


def backtest(daily: pd.DataFrame, weekly: pd.DataFrame, dividends: pd.DataFrame,
             start_date, end_date, settings: Settings) -> Backtest:
    """Signals at weekly close, execute *next trading day's open*.

    Both strategy and buy/hold benchmark get gross cash on dividend ex-date;
    held shares are multiplied by bonus/transfer factor before that day's open.
    Cash credited on ex-date is an accrual approximation, not actual pay date.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    view = daily.loc[(daily.date >= start) & (daily.date <= end)].sort_values("date").copy()
    if view.empty:
        raise ValueError("选择的日期区间内没有交易数据。")
    if settings.initial_cash <= 0:
        raise ValueError("初始资金必须大于0。")
    week_by_date = {pd.Timestamp(r.date): r for _, r in weekly.iterrows()}
    # Need both trade execution and weekly signal strictly within the selected window.
    event_by_day: dict[pd.Timestamp, list] = {}
    for ev in dividends.itertuples(index=False):
        if start <= ev.ex_date <= end:
            event_by_day.setdefault(pd.Timestamp(ev.ex_date), []).append(ev)
    cash = bench_cash = float(settings.initial_cash)
    shares = bench_shares = 0.0
    pending: tuple[str, pd.Timestamp] | None = None
    trades = []
    paid = []
    history = []
    for i, row in enumerate(view.itertuples(index=False)):
        date = pd.Timestamp(row.date)
        # Entitlement is decided on record date before the ex-date. The order
        # executed at ex-date open must not receive this distribution.
        for ev in event_by_day.get(date, []):
            for label, held in (("strategy", shares), ("benchmark", bench_shares)):
                cash_amt = held * float(ev.cash_per_old_share)
                if cash_amt > 1e-8:
                    paid.append({"date": date, "account": label, "shares_before": held,
                                 "cash": cash_amt, "per_share": float(ev.cash_per_old_share)})
                if label == "strategy":
                    cash += cash_amt
                else:
                    bench_cash += cash_amt
            shares *= float(ev.split_factor)
            bench_shares *= float(ev.split_factor)
        if i == 0:
            qty, bench_cash, _, _ = _buy(bench_cash, float(row.open), settings)
            bench_shares = float(qty)
        if pending:
            action, signal_date = pending
            if action == "BUY" and shares < 1e-9:
                qty, new_cash, fee, exec_px = _buy(cash, float(row.open), settings)
                if qty:
                    cash, shares = new_cash, float(qty)
                    trades.append({"signal_date": signal_date, "date": date, "action": "BUY",
                                   "price": exec_px, "shares": float(qty), "fee": fee, "cash_after": cash})
            elif action == "SELL" and shares > 1e-9:
                sold = shares
                cash, fee, exec_px = _sell(cash, shares, float(row.open), settings)
                shares = 0.0
                trades.append({"signal_date": signal_date, "date": date, "action": "SELL",
                               "price": exec_px, "shares": sold, "fee": fee, "cash_after": cash})
            pending = None
        history.append({"date": date, "equity": cash + shares * float(row.close),
                        "benchmark_equity": bench_cash + bench_shares * float(row.close),
                        "cash": cash, "shares": shares, "close": float(row.close)})
        if date in week_by_date:
            buy_signal, sell_signal = conditions(week_by_date[date], settings.strategy)
            if shares < 1e-9 and buy_signal:
                pending = ("BUY", date)
            elif shares > 1e-9 and sell_signal:
                pending = ("SELL", date)
    portfolio = pd.DataFrame(history)
    portfolio["strategy_return_pct"] = (portfolio.equity / settings.initial_cash - 1) * 100
    portfolio["benchmark_return_pct"] = (portfolio.benchmark_equity / settings.initial_cash - 1) * 100
    span_days = max(1, int((portfolio.date.iloc[-1] - portfolio.date.iloc[0]).days))
    final_equity = float(portfolio.equity.iloc[-1])
    bench_equity = float(portfolio.benchmark_equity.iloc[-1])
    annual_factor = 365.25 / span_days
    metrics = {
        "total_return": (final_equity / settings.initial_cash - 1) * 100,
        "benchmark_return": (bench_equity / settings.initial_cash - 1) * 100,
        "annualized_return": ((final_equity / settings.initial_cash) ** annual_factor - 1) * 100,
        "max_drawdown": _mdd(portfolio.equity),
        "benchmark_drawdown": _mdd(portfolio.benchmark_equity),
        "trade_count": len(trades),
        "buy_count": sum(t["action"] == "BUY" for t in trades),
        "sell_count": sum(t["action"] == "SELL" for t in trades),
        "dividend_cash": sum(t["cash"] for t in paid if t["account"] == "strategy"),
        "final_value": final_equity,
        "period_days": span_days,
    }
    return Backtest(portfolio=portfolio, trades=pd.DataFrame(trades),
                    cash_dividends=pd.DataFrame(paid), metrics=metrics)
