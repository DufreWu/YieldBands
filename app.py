"""Run: streamlit run app.py               (local data only)
     streamlit run app.py -- --update-data  (download each requested stock once, then save locally)"""
from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from yieldbands.catalog import STOCKS, display_options
from yieldbands.data import AkshareProvider, DataSourceError, StockData, valid_symbol
from yieldbands.store import LocalStore
from yieldbands.engine import (BOLL_CONDITIONS, DEFAULT_GRID, STRATEGIES, GridRule, Settings,
                               backtest, build_daily, build_weekly)
from yieldbands.watch import snapshot

st.set_page_config(page_title="YieldBands | 股息率 × 布林带", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1450px;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.small-note {color: #667587; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


UPDATE_DATA = "--update-data" in sys.argv[1:]
STORE = LocalStore()
UPDATE_HINT = "用 `streamlit run app.py -- --update-data` 启动可更新数据。"


@st.cache_resource(show_spinner=False)
def update_once(symbol: str) -> None:
    """Download at most once per server process; failures are not cached and retry on rerun."""
    STORE.update(symbol, AkshareProvider())


def load_market_data(symbol: str, start: date, end: date) -> StockData:
    if UPDATE_DATA:
        try:
            update_once(symbol)
        except DataSourceError as exc:
            if not STORE.has(symbol):
                raise
            st.warning(f"{symbol} 数据更新失败，继续使用本地数据：{exc}")
    return STORE.fetch(symbol, start, end)


def data_status(codes: list[str]) -> str:
    times = [t for t in map(STORE.updated_at, codes) if t]
    if not times:
        return "本地暂无数据。" + UPDATE_HINT
    mode = "本次启动会更新数据。" if UPDATE_DATA else UPDATE_HINT
    return f"本地数据下载于 {min(times):%Y-%m-%d %H:%M}。{mode}"


GRID_ACTIONS = {"买": "BUY", "卖": "SELL"}


def grid_editor() -> tuple[GridRule, ...]:
    st.caption("**买**：股息率 ≥ 阈值且满足周线条件 → 投入初始资金的「仓位%」，每档在卖出前只买一次。"
               "**卖**：股息率 ≤ 阈值且满足周线条件 → 卖出当前持仓的「仓位%」，先卖最近买入的档。")
    default = pd.DataFrame([{"动作": "买" if r.action == "BUY" else "卖", "股息率%": r.yield_pct,
                             "周线": r.boll, "仓位%": r.size_pct} for r in DEFAULT_GRID])
    table = st.data_editor(default, key="grid_rules", num_rows="dynamic", hide_index=True, use_container_width=True,
                           column_config={
                               "动作": st.column_config.SelectboxColumn(options=list(GRID_ACTIONS), required=True),
                               "股息率%": st.column_config.NumberColumn(min_value=0.1, max_value=30.0, step=0.1,
                                                                     format="%.1f", required=True),
                               "周线": st.column_config.SelectboxColumn(options=list(BOLL_CONDITIONS), required=True),
                               "仓位%": st.column_config.NumberColumn(min_value=1, max_value=100, step=5,
                                                                    format="%d", required=True),
                           })
    rules = tuple(GridRule(GRID_ACTIONS[r["动作"]], float(r["股息率%"]), r["周线"], float(r["仓位%"]))
                  for r in table.to_dict("records") if all(pd.notna(v) for v in r.values()))
    buy_total = sum(r.size_pct for r in rules if r.action == "BUY")
    if buy_total > 100:
        st.warning(f"买入档合计 {buy_total:g}% 超过100%，资金不足时后面的档按剩余现金买入。")
    return rules


def display_chart(daily: pd.DataFrame, weekly: pd.DataFrame, trades: pd.DataFrame, start: date, end: date,
                  grid: tuple[GridRule, ...] = ()):
    w = weekly[(weekly.date >= pd.Timestamp(start)) & (weekly.date <= pd.Timestamp(end))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.1,
                        subplot_titles=("股价周线（后复权）与 Bollinger Bands", "TTM 股息率与历史均线 / Bollinger Bands"))
    fig.add_trace(go.Scatter(x=w.date, y=w.price, name="周收盘价", line=dict(color="#1d4ed8", width=2)), row=1, col=1)
    for name, label, color, dash in [
        ("price_mid", "价格中轨", "#94a3b8", "dot"),
        ("price_upper", "价格上轨", "#e87958", "dash"),
        ("price_lower", "价格下轨", "#14a389", "dash"),
    ]:
        fig.add_trace(go.Scatter(x=w.date, y=w[name], name=label, line=dict(color=color, dash=dash, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.date, y=w.yield_pct, name="股息率 TTM", line=dict(color="#7c3aed", width=2.4)), row=2, col=1)
    for name, label, color, dash in [
        ("yield_mean", "股息率 X 年均线", "#eab308", "solid"),
        ("yield_pct_upper", "股息率上轨", "#e87958", "dash"),
        ("yield_pct_lower", "股息率下轨", "#14a389", "dash"),
    ]:
        fig.add_trace(go.Scatter(x=w.date, y=w[name], name=label, line=dict(color=color, dash=dash, width=1.6)), row=2, col=1)
    for rule in grid:
        fig.add_hline(y=rule.yield_pct, row=2, col=1, line_dash="dot", line_width=1.2,
                      line_color="#059669" if rule.action == "BUY" else "#dc2626",
                      annotation_text=rule.label(), annotation_position="top left", annotation_font_size=11)
    if not trades.empty:
        signal_to_week = w.set_index("date")
        for act, label, color, marker in [("BUY", "买入（信号周）", "#059669", "triangle-up"),
                                           ("SELL", "卖出（信号周）", "#dc2626", "triangle-down")]:
            selected = trades[trades.action == act]
            days = [t for t in pd.to_datetime(selected.signal_date) if t in signal_to_week.index]
            if days:
                fig.add_trace(go.Scatter(x=days, y=[float(signal_to_week.loc[t, "price"]) for t in days],
                                         mode="markers", name=label, marker=dict(size=13, symbol=marker,
                                                                                color=color, line=dict(width=1.3, color="white"))), row=1, col=1)
    fig.update_layout(height=720, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=-0.12, x=0), margin=dict(t=55, l=45, r=25, b=90))
    fig.update_yaxes(title_text="元 / 股", row=1, col=1)
    fig.update_yaxes(title_text="股息率 %", ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def display_equity(back):
    p = back.portfolio
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.date, y=p.strategy_return_pct, mode="lines", name="策略累计收益", line=dict(color="#2563eb", width=2.8)))
    fig.add_trace(go.Scatter(x=p.date, y=p.benchmark_return_pct, mode="lines", name="买入持有（含现金分红）", line=dict(color="#f59e0b", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b")
    fig.update_layout(height=370, template="plotly_white", hovermode="x unified", yaxis_title="累计收益（%）",
                      legend=dict(orientation="h", y=-0.28), margin=dict(t=15, l=30, r=25, b=65))
    st.plotly_chart(fig, use_container_width=True)


def stock_page():
    st.title("📈 YieldBands")
    st.caption("Dividend Yield × Bollinger Bands · A 股股息率、周线交易信号及历史回测")

    with st.sidebar:
        st.header("1 · 股票和时间")
        known = display_options()
        stock_label = st.selectbox("选择股票", [*known.keys(), "自定义 A 股代码"], index=0)
        code = known.get(stock_label) or st.text_input("六位 A 股代码", value="600036", max_chars=6).strip()
        end_limit = date.today()
        default_start = date(max(2005, end_limit.year - 10), 1, 1)
        selected_dates = st.date_input("回测日期范围", value=(default_start, end_limit),
                                       min_value=date(2000, 1, 1), max_value=end_limit,
                                       format="YYYY-MM-DD")
        st.header("2 · 策略")
        chosen = st.selectbox("买卖策略", list(STRATEGIES), format_func=lambda code: STRATEGIES[code])
        grid = grid_editor() if chosen == "grid" else ()
        x_years = st.slider("股息率历史均线 X（年）", min_value=1, max_value=10, value=3)
        boll_weeks = st.slider("周线 Boll 周期（周）", min_value=5, max_value=60, value=20)
        sigma = st.slider("Boll 标准差倍数", min_value=1.0, max_value=3.5, value=2.0, step=0.25)
        if chosen == "grid":
            st.caption("网格策略中，X 年均线只用于图表参考；Boll 周期和倍数决定「周线」条件的上/中/下轨。")
        else:
            st.caption("默认策略：价格周线触及下轨 + 股息率高于过去 X 年均值 → 买；触及上轨 → 卖。仅持有多头，不做空。")
        st.header("3 · 资金与成本")
        initial_cash = st.number_input("初始资金（元）", min_value=10_000, max_value=100_000_000, value=100_000, step=10_000)
        commission_pct = st.number_input("交易佣金（单边 %，最低5元）", min_value=0.0, max_value=1.0, value=0.03, step=0.01, format="%.2f")
        stamp_pct = st.number_input("卖出印花税（%）", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
        slip_bps = st.number_input("滑点（基点 / 单边）", min_value=0, max_value=100, value=5, step=1)
        st.caption("成本是全区间固定假设，未按历史税费制度逐日调整。")
        run = st.button("加载数据并开始分析", type="primary", use_container_width=True)
        if run:
            st.session_state["analysis_activated"] = True
        st.caption(data_status([code] if valid_symbol(code) else []))

    if not st.session_state.get("analysis_activated", False):
        st.info("左侧选择股票、时间范围及策略，然后点击「加载数据并开始分析」。")
        st.markdown("""
        ### 内置策略
        **① 价格周线下轨买 / 上轨卖 + 股息率过滤（默认）**  
        周线后复权收盘价 ≤ 布林下轨，且当周 TTM 股息率 > 过去 X 年的历史平均股息率时买入；
        已持仓且周线收盘价 ≥ 布林上轨时卖出。

        **② 股息率周线上轨买 / 下轨卖**  
        股息率 ≥ 自身布林上轨且大于 X 年平均值时买入；股息率 ≤ 自身布林下轨时卖出。

        **③ 股息率 X 年均线切换**  
        股息率高于历史均线时买入，低于时卖出（用于对照）。

        **④ 纯价格 Boll 周线策略**  
        不要求股息率高于均值，检验加入股息率条件后是否改善结果。

        **⑤ 网格：股息率分档 + 周线 Boll**  
        自定义多条规则分批买卖，例如：股息率 ≥ 5% 且周线 ≤ 下轨买入 50%；≥ 5.5% 且周线 ≤ 下轨再买 50%；
        股息率 ≤ 4.5% 且周线 ≥ 上轨全部卖出。每个买入档在卖出前只成交一次；部分卖出时先卖最近买入的档。

        **回测模型**：使用周线末个交易日收盘信号，下一交易日开盘价执行；A 股买入手数为100股整数倍；计入手续费、滑点和卖出印花税。将现金分红和送转股纳入策略和买入持有对照。图中买卖点标识的是信号日期，交易表是实际成交日期。
        """)
        st.stop()

    if len(selected_dates) != 2:
        st.error("请选择完整的起止日期。")
        st.stop()
    start_date, end_date = selected_dates
    if start_date >= end_date:
        st.error("结束日期必须晚于开始日期。")
        st.stop()
    if not valid_symbol(code):
        st.error("请输入6位股票代码，例如 600036。")
        st.stop()
    settings = Settings(strategy=chosen, lookback_years=x_years, boll_weeks=boll_weeks,
                        boll_sigma=sigma, initial_cash=float(initial_cash),
                        commission=commission_pct / 100, stamp_tax=stamp_pct / 100,
                        slippage=slip_bps / 10_000, **({"grid": grid} if chosen == "grid" else {}))
    # To create a full X-year weekly average and 1-year TTM yield even on the first
    # selected trading day, retrieve (X + 2) years of warm-up data first.
    fetch_start = max(date(1990, 1, 1), start_date - timedelta(days=(x_years + 2) * 366))
    with st.spinner("正在下载并保存行情与分红记录……" if UPDATE_DATA else "正在读取本地数据……"):
        try:
            stock = load_market_data(code, fetch_start, end_date)
            daily = build_daily(stock.prices, stock.dividends)
            weekly = build_weekly(daily, settings)
            result = backtest(daily, weekly, stock.dividends, start_date, end_date, settings)
        except (DataSourceError, ValueError, KeyError, TypeError, AttributeError) as exc:
            st.error(f"数据或计算失败：{exc}")
            if UPDATE_DATA:
                st.warning("外部数据接口可能暂时不可用。请稍后刷新，或核对 AKShare 版本/网络连接；程序不会用模拟行情冒充真实行情。")
            st.stop()
        except Exception as exc:
            st.error(f"发生未预期错误（{type(exc).__name__}）：{str(exc)[:250]}")
            st.stop()

    st.subheader(f"{STOCKS.get(code, '自定义股票')}（{code}）· {start_date} — {end_date}")
    for msg in stock.warnings:
        st.warning(msg)
    view = daily[(daily.date >= pd.Timestamp(start_date)) & (daily.date <= pd.Timestamp(end_date))]
    if not view.empty and view.date.iloc[0].date() > start_date:
        st.info(f"所选起点早于首个可用交易日，实际回测从 {view.date.iloc[0]:%Y-%m-%d} 开始。")
    if not view.empty and view.date.iloc[-1].date() < end_date:
        st.info(f"数据最新交易日是 {view.date.iloc[-1]:%Y-%m-%d}，并非设置的结束日期。")
    if chosen == "grid" and not view.empty:
        y = view.yield_pct.dropna()
        st.caption(f"回测区间 TTM 股息率：最低 {y.min():.2f}% · 中位 {y.median():.2f}% · 最高 {y.max():.2f}% · "
                   f"最新 {y.iloc[-1]:.2f}%（供设置网格阈值参考）")

    m = result.metrics
    cols = st.columns(5)
    cols[0].metric("策略收益率", f"{m['total_return']:+.2f}%", delta=f"超额 {m['total_return'] - m['benchmark_return']:+.2f} pp")
    cols[1].metric("买入持有收益率", f"{m['benchmark_return']:+.2f}%")
    cols[2].metric("策略最大回撤", f"{m['max_drawdown']:.2f}%")
    cols[3].metric("策略年化收益", f"{m['annualized_return']:+.2f}%")
    cols[4].metric("策略现金股息（税前）", f"¥{m['dividend_cash']:,.0f}")

    st.markdown("### 价格、股息率与买卖信号")
    display_chart(daily, weekly, result.trades, start_date, end_date, grid)

    st.markdown("### 策略与买入持有对比")
    display_equity(result)
    col1, col2 = st.columns(2)
    col1.metric("交易次数", f"{m['trade_count']} 次", help="买入和卖出分别算作一次")
    col2.metric("对照组合最大回撤", f"{m['benchmark_drawdown']:.2f}%")
    if m["trade_count"] == 0:
        if chosen == "grid":
            st.warning("该时间段没有满足网格规则的成交。可以对照下方股息率区间调整阈值，或把周线条件放宽为「≤中轨」/「不限」。")
        else:
            st.warning("该时间段没有满足条件的买入或卖出成交。可以缩短股息率均线 X 年周期、调低布林标准差，或更换策略。")

    st.markdown("### 交易记录与现金分红")
    tab1, tab2, tab3 = st.tabs(["买卖明细", "持仓收到的现金分红", "公司分红事件（已实施）"])
    with tab1:
        if result.trades.empty:
            st.caption("无成交记录。")
        else:
            st.dataframe(result.trades.rename(columns={"signal_date": "信号日期", "date": "成交日期", "action": "方向",
                                                       "rule": "规则", "price": "成交价（元）", "shares": "股数", "fee": "成本（元）",
                                                       "cash_after": "剩余现金（元）", "shares_after": "成交后持股"}),
                         use_container_width=True, hide_index=True)
    with tab2:
        if result.cash_dividends.empty:
            st.caption("此区间内没有持仓收到的现金分红。")
        else:
            st.dataframe(result.cash_dividends.rename(columns={"date": "除息日", "account": "账户", "shares_before": "原持股数",
                                                              "cash": "获得股息（元）", "per_share": "税前每股分红（元）"}),
                         use_container_width=True, hide_index=True)
    with tab3:
        div_view = stock.dividends[(stock.dividends.ex_date >= pd.Timestamp(start_date)) &
                                   (stock.dividends.ex_date <= pd.Timestamp(end_date))]
        st.dataframe(div_view.rename(columns={"ex_date": "除息日", "per10": "每10股派息（税前，元）",
                                              "bonus10": "每10股送转股数", "cash_per_old_share": "旧股每股股息（元）",
                                              "split_factor": "送转股份因子", "announcement": "公告日"}),
                     use_container_width=True, hide_index=True)

    st.markdown("### 导出数据")
    files = st.columns(4)
    files[0].download_button("下载日线 + 股息率 CSV", daily.to_csv(index=False).encode("utf-8-sig"),
                             file_name=f"{code}_daily_yield.csv", mime="text/csv", use_container_width=True)
    files[1].download_button("下载周线 + 信号指标 CSV", weekly.to_csv(index=False).encode("utf-8-sig"),
                             file_name=f"{code}_weekly_indicators.csv", mime="text/csv", use_container_width=True)
    files[2].download_button("下载交易明细 CSV", result.trades.to_csv(index=False).encode("utf-8-sig"),
                             file_name=f"{code}_trades.csv", mime="text/csv", use_container_width=True)
    files[3].download_button("下载回测净值 CSV", result.portfolio.to_csv(index=False).encode("utf-8-sig"),
                             file_name=f"{code}_equity.csv", mime="text/csv", use_container_width=True)

    with st.expander("计算方法、交易时点和风险说明", expanded=False):
        st.markdown("""
        - **历史股息率**：过去365天已除息的税前现金分红合计（如有送转股，换算到当日股本口径）÷ 当日不复权收盘价 ×100%。不是预测股息率、公告后预期分红或年报宣告股息率。
        - **X年历史均线**：过去 52×X 个周末 TTM 股息率的简单平均，严格排除当前周，并要求足量历史样本。
        - **价格布林带**：后复权价格的近 N 个周收盘价均值 ± k×总体标准差；价格指标与真实成交价分别用后复权和不复权行情。
        - **信号与成交**：本周最后一个交易日确认信号，下一个交易日开盘执行；仅持有/空仓两状态，不能融资融券；买入是100股整数倍。
        - **分红回测**：已实施的股息按除权除息日确认现金收益，送转股调整股数；这是股息应收的简化处理，现实现金到账可能滞后；不扣股息税。
        - **成本假设**：佣金最低5元，固定费率、固定卖出印花税、滑点；不模拟历史费率变动、冲击成本、停牌后的特殊流动性及配股。
        - **风险提示**：数据来自第三方公共接口，可能有缺漏或错误。历史分红的每10股单位和除息日期务必与公司公告核对；若长期股本变化、配股等复杂事件存在，回测需进一步校正。本工具仅作研究，不提供投资建议。
        """)


WATCH_CSS = """
<style>
.yb-card {background: #1b2536; color: #e5e9f0; border-radius: 14px; overflow: hidden;
          font-variant-numeric: tabular-nums; font-family: "PingFang SC", "Noto Sans CJK SC", sans-serif;}
.yb-head {background: #121c2e; padding: 22px 26px 16px; display: flex; justify-content: space-between;
          align-items: flex-end; gap: 12px; flex-wrap: wrap; border-bottom: 1px solid #2a3548;}
.yb-title {font-size: 1.9rem; font-weight: 800; color: #f8fafc; line-height: 1.2;}
.yb-title b {color: #fbbf24;}
.yb-meta {color: #94a3b8; font-size: 0.95rem; margin-top: 6px;}
.yb-grid {display: grid; grid-template-columns: 2.2fr 1fr 1.4fr 1.1fr; gap: 8px; align-items: center;}
.yb-cols {padding: 12px 26px; color: #94a3b8; font-weight: 600; background: #18233a;}
.yb-cols div:not(:first-child), .yb-r {text-align: right;}
.yb-row {padding: 16px 26px 12px; border-bottom: 1px solid #2a3548;}
.yb-row:nth-child(even) {background: #1f2a3d;}
.yb-name {font-size: 1.45rem; font-weight: 800; color: #f8fafc;}
.yb-sub {color: #8b98ab; font-size: 0.9rem;}
.yb-px {font-size: 1.4rem; font-weight: 700;}
.yb-up {color: #f87171;} .yb-down {color: #4ade80;} .yb-flat {color: #94a3b8;}
.yb-y {font-size: 1.9rem; font-weight: 800; line-height: 1.15;}
.yb-hi {color: #f87171;} .yb-mid {color: #4ade80;} .yb-lo {color: #fbbf24;}
.yb-static {color: #8b98ab; font-size: 0.9rem;}
.yb-pv {font-size: 1.35rem; font-weight: 800; color: #fbbf24;}
.yb-bar {display: inline-flex; gap: 2px; margin: 2px 0;}
.yb-bar i {width: 8px; height: 4px; background: #3b475c; display: block;}
.yb-bar i.on {background: #f59e0b;}
.yb-pl {font-size: 0.82rem; color: #fbbf24; font-weight: 600;}
.yb-foot {display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap;
          margin-top: 12px; padding-top: 10px; border-top: 1px dashed #2f3b50; font-size: 0.9rem;}
.yb-k {color: #8b98ab; margin-right: 6px;}
.yb-chip {display: inline-block; min-width: 44px; text-align: center; padding: 3px 8px; margin-right: 5px;
          border-radius: 6px; font-weight: 700; background: #3b4558; color: #e5e9f0;}
.yb-chip small {color: #a3aec0; font-weight: 400; margin-right: 3px;}
.yb-chip.lo {background: #2f5a35; color: #86efac;} .yb-chip.hi {background: #6b2b2b; color: #fca5a5;}
.yb-sig {display: inline-block; padding: 3px 8px; border-radius: 6px; font-weight: 700;}
.yb-sig.buy {background: #5c4813; color: #fde68a;} .yb-sig.sell {background: #3b4558; color: #cbd5e1;}
.yb-band b {font-weight: 700;} .yb-band .a {color: #fbbf24;} .yb-band .g {color: #4ade80;}
.yb-band small {color: #8b98ab; margin-right: 6px;}
.yb-note {padding: 14px 26px 18px; color: #8b98ab; font-size: 0.85rem; background: #121c2e;}
@media (max-width: 720px) {
  .yb-head, .yb-cols, .yb-row, .yb-note {padding-left: 14px; padding-right: 14px;}
  .yb-grid {grid-template-columns: 1.6fr 1fr; row-gap: 10px;}
  .yb-cols {display: none;}
  .yb-title {font-size: 1.4rem;}
}
</style>
"""
WEEKDAYS = "一二三四五六日"


def _watch_row(code: str, profile: dict[str, str], snap) -> str:
    name = escape(profile.get("name") or STOCKS.get(code) or code)
    sub = " · ".join(escape(x) for x in (profile.get("industry"), code) if x)
    chg = snap.change_pct
    chg_cls, arrow = ("yb-up", "▲") if chg > 0.005 else ("yb-down", "▼") if chg < -0.005 else ("yb-flat", "")
    y = snap.ttm_yield
    y_cls = "yb-hi" if y >= 6 else "yb-mid" if y >= 5 else "yb-lo"
    if snap.percentile is None:
        pct_html = '<div class="yb-pv">—</div><div class="yb-pl">历史不足</div>'
    else:
        p = round(snap.percentile)  # label must agree with the displayed integer
        level = "高位" if p >= 80 else "低位" if p <= 20 else "中位"
        bar = "".join(f'<i class="{"on" if k < round(p / 10) else ""}"></i>' for k in range(10))
        pct_html = (f'<div class="yb-pv">{p:.0f}%</div><div class="yb-bar">{bar}</div>'
                    f'<div class="yb-pl">{snap.percentile_years:.0f}年{level}</div>')
    chips = "".join(
        f'<span class="yb-chip {"lo" if pos in ("下", "中下") else "hi" if pos in ("上", "中上") else ""}" '
        f'title="{frame}线 Boll"><small>{frame}</small>{pos or "—"}</span>'
        for frame, pos in snap.boll.items())
    if snap.signal:
        chips += f'<span class="yb-sig {"buy" if snap.signal == "买入信号" else "sell"}">⭐ {snap.signal}</span>'
    band = ""
    if snap.targets:
        (y1, p1), (y2, p2) = snap.targets
        band = (f'<span class="yb-band"><span class="yb-k">股息率价格区间</span>'
                f'<b class="a">{p1:.2f}</b><small>({y1:g}%)</small>'
                f'<b class="g">{p2:.2f}</b><small>({y2:g}%)</small></span>')
    return f"""
<div class="yb-row">
  <div class="yb-grid">
    <div><div class="yb-name">{name}</div><div class="yb-sub">{sub}</div></div>
    <div class="yb-r"><div class="yb-px">{snap.close:.2f}</div><div class="{chg_cls}">{arrow} {abs(chg):.2f}%</div></div>
    <div class="yb-r"><div class="yb-y {y_cls}">{y:.2f}%</div><div class="yb-static">静态 {snap.static_yield:.2f}%</div></div>
    <div class="yb-r">{pct_html}</div>
  </div>
  <div class="yb-foot"><div><span class="yb-k">BOLL</span>{chips}</div>{band}</div>
</div>"""


def watch_page():
    controls = st.columns([4, 1])
    codes_text = controls[0].text_input("观察列表（6位A股代码，空格或逗号分隔）",
                                        value=st.session_state.get("watch_codes", " ".join(STOCKS)))
    years = controls[1].selectbox("分位窗口", [3, 5, 10], index=2, format_func=lambda y: f"近{y}年")
    st.session_state["watch_codes"] = codes_text
    codes = list(dict.fromkeys(c for c in re.split(r"[\s,，、;；]+", codes_text) if c))
    bad = [c for c in codes if not valid_symbol(c)]
    if bad:
        st.warning(f"已忽略无效代码：{'、'.join(bad)}")
    codes = [c for c in codes if valid_symbol(c)]
    if not codes:
        st.info("请输入至少一个6位A股代码，例如 600036。")
        st.stop()

    settings = Settings()
    end = date.today()
    start = end - timedelta(days=(max(years, settings.lookback_years) + 2) * 366)
    rows, failures = [], []
    verb = "下载" if UPDATE_DATA else "读取"
    progress = st.progress(0.0, text=f"正在{verb}行情与分红……")
    for k, code in enumerate(codes):
        progress.progress(k / len(codes), text=f"正在{verb} {STOCKS.get(code, code)}（{k + 1}/{len(codes)}）……")
        try:
            stock = load_market_data(code, start, end)
            snap = snapshot(build_daily(stock.prices, stock.dividends), stock.dividends, settings, years)
            rows.append((code, STORE.profile(code), snap))
        except Exception as exc:
            failures.append(f"{code}：{str(exc)[:160]}")
    progress.empty()
    for msg in failures:
        st.warning(f"数据读取失败 · {msg}")
    if not rows:
        st.stop()

    rows.sort(key=lambda r: r[2].ttm_yield, reverse=True)
    as_of = max(r[2].date for r in rows)
    stale = [STOCKS.get(c, c) for c, _, s in rows if s.date < as_of]
    header = f"""
<div class="yb-head">
  <div><div class="yb-title"><b>{as_of.month}月{as_of.day}日</b> 股息率观察</div>
  <div class="yb-meta">周{WEEKDAYS[as_of.weekday()]} · 收盘数据</div></div>
  <div class="yb-meta">{len(rows)}只 · 按股息率降序</div>
</div>
<div class="yb-grid yb-cols"><div>公司</div><div>现价</div><div>TTM股息率</div><div>股息率分位</div></div>"""
    note = ('<div class="yb-note">TTM股息率 = 过去365天已实施现金分红 ÷ 现价；静态 = 上一自然年已实施分红 ÷ 现价；'
            f'分位 = 当前TTM股息率在近{years}年日度序列中的百分位；BOLL = 后复权价在日/周/月线 '
            f'Boll({settings.boll_weeks}, {settings.boll_sigma:g}σ) 中的位置；⭐ = 默认策略本周信号；'
            '价格区间 = 达到对应股息率所需股价。仅供参考，不构成任何操作建议。</div>')
    st.html(WATCH_CSS + f'<div class="yb-card">{header}{"".join(_watch_row(*r) for r in rows)}{note}</div>')
    if stale:
        st.caption(f"以下股票最新交易日早于 {as_of:%Y-%m-%d}（可能停牌）：{'、'.join(stale)}")
    st.caption(data_status([c for c, _, _ in rows]))


page = st.navigation([
    st.Page(stock_page, title="个股分析", icon="📈", url_path="stock", default=True),
    st.Page(watch_page, title="股息率观察", icon="💰", url_path="watch"),
], position="top")
page.run()
