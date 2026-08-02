"""Streamlit Backtest 分頁（含歷史存檔重睇）。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.binance import fetch_klines_hours
from src.indicators.core import drop_incomplete_candle
from src.strategy.backtest import BacktestResult, run_backtest
from src.ui.backtest_store import (
    STORE_PATH,
    delete_run,
    dict_to_result,
    get_run,
    list_runs,
    save_run,
)


def _equity_curve(result: BacktestResult) -> pd.DataFrame:
    rows = []
    eq = 0.0
    for t in result.trade_list:
        eq += float(t["pnl_usdt"])
        rows.append({"time": t["exit_time"], "equity": eq, "pnl": t["pnl_usdt"]})
    if not rows:
        return pd.DataFrame(columns=["time", "equity", "pnl"])
    return pd.DataFrame(rows)


def _render_result(result: BacktestResult, *, key_prefix: str = "bt") -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", result.trades)
    m2.metric("Win rate", f"{result.win_rate:.1f}%")
    m3.metric("Net PnL", f"{result.net_pnl_usdt:,.2f}")
    m4.metric("Max DD", f"{result.max_drawdown_usdt:,.2f}")
    m5.metric("Avg PnL%", f"{result.avg_pnl_pct:.3f}%")

    st.caption(
        f"{result.symbol} · {result.start} → {result.end} · bars={result.bars} · "
        f"fees={result.fees_usdt:,.2f} · gross={result.gross_pnl_usdt:,.2f} · label={result.label}"
    )

    if result.by_type:
        st.write(
            "按類型",
            {k: {"筆數": v["n"], "PnL": round(v["pnl"], 2)} for k, v in result.by_type.items()},
        )

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
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_equity")

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
            key=f"{key_prefix}_download",
        )
    else:
        st.warning("呢段時間冇產生交易（可能參數太嚴／warmup 太長）。")


def _render_history_picker() -> None:
    st.markdown("##### 歷史結果")
    st.caption(f"存檔位置：`{STORE_PATH}`（最多保留最近約 40 次）")

    runs = list_runs()
    if not runs:
        st.info("未有已存檔嘅 backtest。跑完一次會自動出現喺呢度。")
        return

    labels = [
        f"{r['saved_at_iso']} · {r['symbol']} · trades={r['trades']} · "
        f"WR={float(r['win_rate'] or 0):.1f}% · net={float(r['net_pnl_usdt'] or 0):.2f} · id={r['id']}"
        for r in runs
    ]
    choice = st.selectbox("揀一次歷史回測重睇", labels, key="bt_hist_select")
    idx = labels.index(choice)
    run_id = runs[idx]["id"]

    col_a, col_b = st.columns(2)
    load = col_a.button("載入呢次結果", key="bt_hist_load")
    delete = col_b.button("刪除呢次", key="bt_hist_del")

    if delete:
        if delete_run(run_id):
            st.success(f"已刪除 {run_id}")
            if st.session_state.get("bt_view_id") == run_id:
                st.session_state.pop("bt_last_result", None)
                st.session_state.pop("bt_view_id", None)
            st.rerun()
        else:
            st.error("刪除失敗")

    if load:
        payload = get_run(run_id)
        if payload is None:
            st.error("搵唔到呢次結果")
            return
        st.session_state.bt_last_result = dict_to_result(payload)
        st.session_state.bt_view_id = run_id
        st.session_state.bt_view_meta = payload.get("meta") or {}
        st.rerun()


def render_backtest_tab(cfg: dict, *, default_notional: float = 1000.0) -> None:
    st.subheader("策略 Backtest")
    st.caption(
        "用而家側邊欄參數，對歷史 5m（已收線）逐根回測。"
        "跑完會自動存檔，之後可喺下面歷史列表重睇。"
    )

    _render_history_picker()
    st.divider()

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

    note = st.text_input("備註（可選）", "", key="bt_note", placeholder="例如：放寬 range / 嚴 SL")
    run = st.button("開始 Backtest", type="primary", key="bt_run")

    if run:
        symbol = cfg.get("symbol", "BTCUSDT")
        hours = int(days) * 24 + 12
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
            entry = save_run(
                result,
                meta={
                    "days": int(days),
                    "notional": float(notional),
                    "warmup": int(warmup),
                    "max_hold": int(max_hold),
                    "note": note,
                    "symbol": symbol,
                    "min_touches": cfg.get("range", {}).get("min_touches"),
                    "touch_tol": cfg.get("range", {}).get("touch_tolerance_pct"),
                    "sl_atr_mult": cfg.get("range", {}).get("sl_atr_mult"),
                },
            )
        st.session_state.bt_last_result = result
        st.session_state.bt_view_id = entry["id"]
        st.session_state.bt_view_meta = entry.get("meta") or {}
        st.success(f"已存檔 · id={entry['id']} · {entry['saved_at_iso']}")

    if "bt_last_result" not in st.session_state:
        st.info("調好左側策略參數 → 撳「開始 Backtest」，或由上面歷史列表載入舊結果。")
        return

    st.divider()
    st.markdown("##### 而家顯示緊")
    view_id = st.session_state.get("bt_view_id")
    meta = st.session_state.get("bt_view_meta") or {}
    if view_id:
        st.caption(
            f"id={view_id}"
            + (f" · 備註：{meta.get('note')}" if meta.get("note") else "")
            + (
                f" · days={meta.get('days')} warmup={meta.get('warmup')} hold={meta.get('max_hold')}"
                if meta
                else ""
            )
        )
    _render_result(st.session_state.bt_last_result, key_prefix=f"view_{view_id or 'session'}")
