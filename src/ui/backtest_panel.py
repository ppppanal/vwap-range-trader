"""Streamlit Backtest 分頁。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.binance import fetch_klines_hours
from src.indicators.core import drop_incomplete_candle
from src.strategy.backtest import BacktestResult, run_backtest


def _equity_curve(result: BacktestResult) -> pd.DataFrame:
    rows = []
    eq = 0.0
    for t in result.trade_list:
        eq += float(t["pnl_usdt"])
        rows.append({"time": t["exit_time"], "equity": eq, "pnl": t["pnl_usdt"]})
    if not rows:
        return pd.DataFrame(columns=["time", "equity", "pnl"])
    return pd.DataFrame(rows)


def render_backtest_tab(cfg: dict, *, default_notional: float = 1000.0) -> None:
    st.subheader("策略 Backtest")
    st.caption(
        "用而家側邊欄參數，對歷史 5m（已收線）逐根回測。"
        "同一時間最多一倉；打到 SL/TP 或超過持倉根數就平倉。"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        days = st.slider("回測日數", 3, 14, 7, 1, key="bt_days")
    with c2:
        notional = st.number_input(
            "基礎名義 (USDT)", 10.0, 100000.0, float(default_notional), 50.0, key="bt_notional"
        )
    with c3:
        warmup = st.number_input("Warmup 根數", 100, 800, 300, 20, key="bt_warmup")
    with c4:
        max_hold = st.number_input("最長持倉（根/5m）", 6, 288, 72, 6, key="bt_max_hold")

    run = st.button("開始 Backtest", type="primary", key="bt_run")

    if not run and "bt_last_result" not in st.session_state:
        st.info("調好左側策略參數 → 撳「開始 Backtest」。結果會用目前參數，唔使改 YAML。")
        return

    if run:
        symbol = cfg.get("symbol", "BTCUSDT")
        hours = int(days) * 24 + 12  # 多留 buffer 畀 warmup
        with st.spinner(f"抓取 {symbol} 近 {days} 日 5m 資料並回測…"):
            raw = fetch_klines_hours(symbol=symbol, interval="5m", hours=hours)
            raw = drop_incomplete_candle(raw, interval_minutes=5)
            result = run_backtest(
                raw,
                cfg,
                label=f"ui_{days}d",
                base_notional=float(notional),
                warmup_bars=int(warmup),
                max_hold_bars=int(max_hold),
            )
        st.session_state.bt_last_result = result

    result: BacktestResult = st.session_state.bt_last_result

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", result.trades)
    m2.metric("Win rate", f"{result.win_rate:.1f}%")
    m3.metric("Net PnL", f"{result.net_pnl_usdt:,.2f}")
    m4.metric("Max DD", f"{result.max_drawdown_usdt:,.2f}")
    m5.metric("Avg PnL%", f"{result.avg_pnl_pct:.3f}%")

    st.caption(
        f"{result.symbol} · {result.start} → {result.end} · bars={result.bars} · "
        f"fees={result.fees_usdt:,.2f} · gross={result.gross_pnl_usdt:,.2f}"
    )

    if result.by_type:
        st.write("按類型", {k: {"筆數": v["n"], "PnL": round(v["pnl"], 2)} for k, v in result.by_type.items()})

    eq = _equity_curve(result)
    if not eq.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eq["time"],
                y=eq["equity"],
                mode="lines",
                name="累計淨 PnL (USDT)",
                line=dict(color="#60a5fa", width=2),
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=30, r=20, t=30, b=30),
            template="plotly_dark",
            title="Equity curve（累計淨利）",
        )
        st.plotly_chart(fig, use_container_width=True)

    if result.trade_list:
        trades_df = pd.DataFrame(result.trade_list)
        show_cols = [
            c
            for c in [
                "entry_time",
                "exit_time",
                "side",
                "trade_type",
                "entry",
                "exit",
                "stop_loss",
                "take_profit",
                "size_mult",
                "pnl_pct",
                "pnl_usdt",
                "exit_reason",
                "reason",
            ]
            if c in trades_df.columns
        ]
        st.dataframe(trades_df[show_cols], use_container_width=True, height=360)
        st.download_button(
            "下載 trades CSV",
            data=trades_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"backtest_{result.label}.csv",
            mime="text/csv",
            key="bt_download",
        )
    else:
        st.warning("呢段時間冇產生交易（可能參數太嚴／warmup 太長）。")
