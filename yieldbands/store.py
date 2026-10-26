"""Local CSV copy of downloaded market data, so normal runs work offline.

Layout: data/<symbol>/prices.csv, dividends.csv, profile.json. The full history
is always stored; callers slice the date range they need.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .data import AkshareProvider, DataSourceError, StockData, finalize, valid_symbol

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_START = date(1990, 1, 1)


class LocalStore:
    def __init__(self, root: Path = DATA_DIR):
        self.root = Path(root)

    def _dir(self, symbol: str) -> Path:
        if not valid_symbol(symbol):
            raise DataSourceError("A股代码需为6位数字，例如 600036。")
        return self.root / symbol

    def has(self, symbol: str) -> bool:
        folder = self._dir(symbol)
        return (folder / "prices.csv").exists() and (folder / "dividends.csv").exists()

    def updated_at(self, symbol: str) -> datetime | None:
        path = self._dir(symbol) / "prices.csv"
        return datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None

    def update(self, symbol: str, provider: AkshareProvider | None = None) -> None:
        """Download the full history and replace the local copy."""
        provider = provider or AkshareProvider()
        stock = provider.fetch(symbol, HISTORY_START, date.today())
        folder = self._dir(symbol)
        folder.mkdir(parents=True, exist_ok=True)
        self._write(folder / "dividends.csv", stock.dividends.to_csv(index=False))
        self._write(folder / "prices.csv", stock.prices.to_csv(index=False))
        profile = provider.profile(symbol)
        if profile or not (folder / "profile.json").exists():
            self._write(folder / "profile.json", json.dumps(profile, ensure_ascii=False))

    def fetch(self, symbol: str, start: date, end: date) -> StockData:
        if not self.has(symbol):
            raise DataSourceError(f"本地没有 {symbol} 的数据。请用 `streamlit run app.py -- --update-data` 启动以下载。")
        folder = self._dir(symbol)
        prices = pd.read_csv(folder / "prices.csv", parse_dates=["date"])
        dividends = pd.read_csv(folder / "dividends.csv", parse_dates=["ex_date", "announcement"])
        prices = prices[(prices.date >= pd.Timestamp(start)) & (prices.date <= pd.Timestamp(end))]
        if prices.empty:
            raise DataSourceError(f"本地 {symbol} 数据在 {start} — {end} 之间没有行情。")
        return finalize(prices.reset_index(drop=True), dividends)

    def profile(self, symbol: str) -> dict[str, str]:
        path = self._dir(symbol) / "profile.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
