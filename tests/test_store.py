import pandas as pd
import pytest

from yieldbands.data import DataSourceError, finalize
from yieldbands.store import LocalStore


class FakeProvider:
    calls = 0

    def fetch(self, symbol, start, end):
        self.calls += 1
        days = pd.bdate_range("2020-01-01", "2021-12-31")
        prices = pd.DataFrame({"date": days, "open": 10.0, "close": 10.0, "volume": 1.0, "adj_close": 10.0})
        div = pd.DataFrame([{"ex_date": pd.Timestamp("2020-06-10"), "per10": 5.0, "bonus10": 0.0,
                             "cash_per_old_share": 0.5, "split_factor": 1.0, "announcement": pd.NaT}])
        return finalize(prices, div)

    def profile(self, symbol):
        return {"name": "测试股份", "industry": "测试业"}


def test_missing_local_data_explains_update_flag(tmp_path):
    with pytest.raises(DataSourceError, match="--update-data"):
        LocalStore(tmp_path).fetch("600036", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2021-01-01").date())


def test_update_then_read_slices_locally(tmp_path):
    store, provider = LocalStore(tmp_path), FakeProvider()
    store.update("600036", provider)
    assert store.has("600036") and store.updated_at("600036") is not None
    stock = store.fetch("600036", pd.Timestamp("2021-01-01").date(), pd.Timestamp("2021-06-30").date())
    assert stock.prices.date.min() == pd.Timestamp("2021-01-01")
    assert stock.prices.date.max() == pd.Timestamp("2021-06-30")
    assert stock.dividends.cash_per_old_share.tolist() == [0.5]  # earlier dividends kept for TTM
    assert store.profile("600036") == {"name": "测试股份", "industry": "测试业"}
    assert provider.calls == 1
