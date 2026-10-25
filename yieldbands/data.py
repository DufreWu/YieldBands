"""Historical stock prices and implemented cash distributions, via AKShare.

All dividend cash amounts and bonus-share figures from the Sina/AKShare interface
are per *10 old shares*. The interface and its upstream source can change; users
should reconcile important payouts against exchange announcements.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd


class DataSourceError(RuntimeError):
    pass


@dataclass
class StockData:
    prices: pd.DataFrame  # date, open, close, adj_close, ...
    dividends: pd.DataFrame  # ex_date, per10, bonus10, cash_per_old_share, split_factor
    warnings: list[str]


def valid_symbol(symbol: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", str(symbol)))


def _get_column(df: pd.DataFrame, choices: tuple[str, ...], *, required=True) -> str | None:
    for name in choices:
        if name in df.columns:
            return name
    if required:
        raise DataSourceError(f"数据缺少必需字段 {choices}；实际字段: {list(df.columns)}")
    return None


def _numeric(series: pd.Series, default=0.0) -> pd.Series:
    # '--', empty strings, commas and thousands separators may appear upstream.
    txt = series.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(txt.replace({"--": None, "nan": None, "": None, "None": None}), errors="coerce").fillna(default)


def normalize_price(raw: pd.DataFrame, hfq: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or hfq.empty:
        raise DataSourceError("历史行情为空：可能是网络、接口限流或股票代码问题。")
    c_date = _get_column(raw, ("日期", "date", "Date"))
    c_open = _get_column(raw, ("开盘", "open", "Open"))
    c_close = _get_column(raw, ("收盘", "close", "Close"))
    c_volume = _get_column(raw, ("成交量", "volume", "Volume"), required=False)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[c_date], errors="coerce"),
        "open": _numeric(raw[c_open], default=float("nan")),
        "close": _numeric(raw[c_close], default=float("nan")),
    })
    if c_volume:
        out["volume"] = _numeric(raw[c_volume])
    ca_date = _get_column(hfq, ("日期", "date", "Date"))
    ca_close = _get_column(hfq, ("收盘", "close", "Close"))
    adjusted = pd.DataFrame({
        "date": pd.to_datetime(hfq[ca_date], errors="coerce"),
        "adj_close": _numeric(hfq[ca_close], default=float("nan")),
    })
    out = out.merge(adjusted, on="date", how="inner")
    out = out.dropna(subset=["date", "open", "close", "adj_close"])
    out = out.loc[(out[["open", "close", "adj_close"]] > 0).all(axis=1)]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if out.empty:
        raise DataSourceError("无法将不复权/后复权行情按交易日合并。")
    return out


def normalize_dividends(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = ["ex_date", "per10", "bonus10", "cash_per_old_share", "split_factor", "announcement"]
    if raw.empty:
        raise DataSourceError("分红接口返回空表，停止回测以避免把缺失分红误判为零股息。")
    c_ex = _get_column(raw, ("除权除息日", "除息日", "ex_date"))
    c_cash = _get_column(raw, ("派息(税前)(元)", "派息(税前)", "派息", "每10股派息(元)", "per10"))
    c_bonus = _get_column(raw, ("送股(股)", "送股", "送股比例", "bonus"), required=False)
    c_transfer = _get_column(raw, ("转增(股)", "转增", "转增比例", "transfer"), required=False)
    c_ann = _get_column(raw, ("公告日期", "announcement"), required=False)
    c_state = _get_column(raw, ("进度", "状态", "status"), required=False)
    frame = pd.DataFrame({
        "ex_date": pd.to_datetime(raw[c_ex], errors="coerce"),
        "per10": _numeric(raw[c_cash]),
        "bonus10": (_numeric(raw[c_bonus]) if c_bonus else pd.Series(0.0, index=raw.index))
                  + (_numeric(raw[c_transfer]) if c_transfer else pd.Series(0.0, index=raw.index)),
        "announcement": (pd.to_datetime(raw[c_ann], errors="coerce") if c_ann
                         else pd.Series(pd.NaT, index=raw.index)),
    })
    if c_state:
        state = raw[c_state].astype(str)
        frame = frame.loc[state.str.contains("实施|完成|已执行", regex=True, na=False)].copy()
    frame = frame.loc[frame.ex_date.notna()].copy()  # no planned or future undated dividends
    frame = frame.loc[(frame.per10 > 0) | (frame.bonus10 > 0)].copy()
    if (frame[["per10", "bonus10"]]<0).any().any() or (frame.bonus10 >= 100).any():
        raise DataSourceError("分红/送转数值存在异常，请核验来源字段单位。")
    # Drop exact duplicates, but preserve legitimately distinct same-day events.
    frame = frame.drop_duplicates(subset=["ex_date", "per10", "bonus10"]).sort_values("ex_date")
    frame["cash_per_old_share"] = frame["per10"] / 10.0
    frame["split_factor"] = 1.0 + frame["bonus10"] / 10.0
    frame = frame[columns].reset_index(drop=True)
    if frame.empty:
        raise DataSourceError("未发现已实施且有除权除息日的分红记录；无法构建历史股息率。")
    warns=[]
    if frame.ex_date.duplicated().any():
        warns.append("发现同日多条权益事件；请核实是否重复，分红已逐笔处理。")
    return frame, warns


class AkshareProvider:
    """Provider boundary: replace this class to add more data sources."""

    def fetch(self, symbol: str, start_date: date, end_date: date) -> StockData:
        if not valid_symbol(symbol):
            raise DataSourceError("A股代码需为6位数字，例如 600036。")
        try:
            import akshare as ak
        except ImportError as exc:
            raise DataSourceError("请先安装 requirements.txt 中的 akshare。") from exc
        params = dict(symbol=symbol, period="daily", start_date=start_date.strftime("%Y%m%d"),
                      end_date=end_date.strftime("%Y%m%d"))
        try:
            raw = ak.stock_zh_a_hist(**params, adjust="")
            hfq = ak.stock_zh_a_hist(**params, adjust="hfq")
        except Exception as exc:
            raise DataSourceError(f"股票行情抓取失败：{str(exc)[:220]}") from exc
        prices = normalize_price(raw, hfq)
        dividend_fn = ak.stock_history_dividend_detail
        # Some versions of AKShare use stock=, others use symbol=.
        sig = inspect.signature(dividend_fn)
        kwargs = {"indicator": "分红"}
        kwargs["symbol" if "symbol" in sig.parameters else "stock"] = symbol
        if "date" in sig.parameters:
            kwargs["date"] = ""
        try:
            raw_div = dividend_fn(**kwargs)
        except Exception as exc:
            raise DataSourceError(f"分红历史抓取失败：{str(exc)[:220]}") from exc
        dividends, warnings = normalize_dividends(raw_div)
        # Only keep ex-dates up to the end of returned data; announced future corporate
        # actions cannot participate in backtests.
        dividends = dividends[dividends.ex_date <= prices.date.max()].copy()
        if dividends.empty:
            raise DataSourceError("当前行情时间段内没有可用的已实施分红历史。")
        if len(prices) < 100:
            warnings.append("获取到的行情少于100个交易日，回测可能无法生成信号。")
        if not set(dividends.ex_date).intersection(set(prices.date)):
            warnings.append("下载的价格区间内没有除息日；请检查股息率历史窗口或扩大范围。")
        return StockData(prices=prices, dividends=dividends.reset_index(drop=True), warnings=warnings)
