"""VWAP Range Trader — Streamlit 控制台（參數跨使用者同步）。"""

from __future__ import annotations

import copy
import hashlib
import os
import time
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

import pandas as pd

from src.data.binance import LAST_SOURCE, fetch_klines_hours, fetch_ticker_24h
from src.indicators.core import compute_indicators, drop_incomplete_candle
from src.strategy.engine import StrategyEngine
from src.ui.backtest_panel import render_backtest_tab
from src.ui.shared_params import PARAM_KEYS, default_params, load_shared, save_shared

# 顯示用：Binance BTCUSDT ≈ BTC/USD
SYMBOL_LABELS = {
    "BTCUSDT": "BTC/USD",
    "ETHUSDT": "ETH/USD",
    "SOLUSDT": "SOL/USD",
}

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = ROOT / "config" / "default.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_default_cfg() -> dict:
    with open(DEFAULT_CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_ui_cfg(base: dict, ui: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["symbol"] = ui["symbol"]
    cfg["lookback_hours"] = ui["lookback_hours"]
    cfg["refresh_seconds"] = ui["refresh_seconds"]
    cfg["vwap"]["rolling_hours"] = ui["vwap_hours"]
    cfg["strength"]["size_near_pct"] = ui["size_near_pct"]
    cfg["strength"]["size_mid_pct"] = ui["size_mid_pct"]
    cfg["strength"]["size_far_pct"] = ui["size_far_pct"]
    cfg["intercept"]["max_4h"] = ui["max_intercept_4h"]
    cfg["intercept"]["max_6h"] = ui["max_intercept_6h"]
    cfg["range"]["touch_tolerance_pct"] = ui["touch_tol"]
    cfg["range"]["min_touches"] = ui["min_touches"]
    cfg["range"]["entry_buffer_pct"] = ui["entry_buffer_pct"]
    cfg["range"]["min_range_fee_mult"] = ui["min_range_fee_mult"]
    cfg["range"]["fee_rate"] = ui["fee_rate"]
    cfg["range"]["sl_atr_mult"] = ui["sl_atr_mult"]
    cfg["range"]["atr_period"] = ui["atr_period"]
    cfg["volume"]["min_btc_threshold"] = ui["min_btc_vol"]
    cfg["volume"]["avg_period"] = ui["vol_avg_period"]
    cfg["volume"]["low_vol_consecutive"] = ui["low_vol_consec"]
    cfg["volume"]["breakout_max_vol_mult"] = ui["breakout_vol_mult"]
    cfg["breakout"]["trailing_enabled"] = ui["trailing_enabled"]
    return cfg


def build_chart(df, state) -> go.Figure:
    # 圖只顯示最近 24h（288 根已收線 5m）
    show = df.iloc[-min(len(df), 288) :].copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=("5m 已收線 · VWAP · Range（突破≥5次線位）", "成交量 (BTC)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=show.index,
            open=show["open"],
            high=show["high"],
            low=show["low"],
            close=show["close"],
            name="5m closed",
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=show.index,
            y=show["vwap"],
            name="24h VWAP",
            line=dict(color="#f39c12", width=2),
        ),
        row=1,
        col=1,
    )
    if state.range_high is not None:
        fig.add_hline(
            y=state.range_high,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Range High ×{state.high_touches}",
            row=1,
            col=1,
        )
    if state.range_low is not None:
        fig.add_hline(
            y=state.range_low,
            line_dash="dash",
            line_color="#27ae60",
            annotation_text=f"Range Low ×{state.low_touches}",
            row=1,
            col=1,
        )

    sig = state.signal
    if sig.entry is not None:
        fig.add_hline(y=sig.entry, line_dash="dot", line_color="#3498db", annotation_text=f"Entry {sig.side}", row=1, col=1)
    if sig.stop_loss is not None:
        fig.add_hline(y=sig.stop_loss, line_dash="dot", line_color="#c0392b", annotation_text="SL", row=1, col=1)
    if sig.take_profit is not None:
        fig.add_hline(y=sig.take_profit, line_dash="dot", line_color="#16a085", annotation_text="TP", row=1, col=1)

    colors = ["#2ecc71" if c >= o else "#e74c3c" for o, c in zip(show["open"], show["close"])]
    fig.add_trace(go.Bar(x=show.index, y=show["volume"], name="Volume", marker_color=colors), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=show.index, y=show["vol_avg"], name="Vol Avg(20)", line=dict(color="#9b59b6", width=1.5)),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=640,
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(type="date", row=1, col=1)
    return fig


def _fmt_px(v) -> str:
    if v is None:
        return "—"
    return f"{float(v):,.2f}"


def render_signal_panel(state, suggested_size: float) -> None:
    """右邊訊號可視化面板。"""
    sig = state.signal
    side = sig.side or "flat"
    side_meta = {
        "long": ("做多 LONG", "#1e8449", "#d5f5e3"),
        "short": ("做空 SHORT", "#922b21", "#fadbd8"),
        "flat": ("觀望 FLAT", "#1a5276", "#d4e6f1"),
    }
    title, accent, soft = side_meta.get(side, side_meta["flat"])
    trade_type = (sig.trade_type or state.mode or "—").upper()

    rr = "—"
    risk = None
    reward = None
    if sig.entry is not None and sig.stop_loss is not None and sig.take_profit is not None:
        risk = abs(float(sig.entry) - float(sig.stop_loss))
        reward = abs(float(sig.take_profit) - float(sig.entry))
        if risk > 0:
            rr = f"{reward / risk:.2f}R"

    st.markdown(
        f"""
<div style="border:1px solid {accent}; background:linear-gradient(180deg,{soft}22,#111827); border-radius:12px; padding:14px 16px; margin-bottom:12px;">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
    <div style="font-size:1.35rem; font-weight:800; color:{accent};">{title}</div>
    <div style="background:{accent}; color:#fff; padding:4px 10px; border-radius:999px; font-size:0.8rem; font-weight:700;">{trade_type}</div>
  </div>
  <div style="color:#9ca3af; margin-top:6px; font-size:0.9rem;">{sig.reason or "等候觸發條件"}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Entry", _fmt_px(sig.entry))
    m2.metric("Stop Loss", _fmt_px(sig.stop_loss))
    m3.metric("Take Profit", _fmt_px(sig.take_profit))

    m4, m5, m6 = st.columns(3)
    m4.metric("Trailing SL", _fmt_px(sig.trailing_sl))
    m5.metric("倉位倍數", f"{float(sig.size_mult or 0):.2f}×")
    m6.metric("建議名義", f"{suggested_size:,.0f}")

    if risk is not None and reward is not None:
        st.progress(min(1.0, reward / (risk + reward) if (risk + reward) > 0 else 0.0), text=f"風險回報 · {rr}")

    st.markdown("##### Range 線位（24h 突破≥5次）")
    ok = "✅" if state.range_valid and state.range_vs_fee_ok else "⏳" if state.range_valid else "❌"
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(
            f"""
<div style="background:#1f2937;border-radius:10px;padding:12px;border-left:4px solid #e74c3c;">
  <div style="color:#f87171;font-size:0.8rem;">RANGE HIGH</div>
  <div style="font-size:1.3rem;font-weight:700;color:#fff;">{_fmt_px(state.range_high)}</div>
  <div style="color:#9ca3af;font-size:0.8rem;">突破 {state.high_touches} 次</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"""
<div style="background:#1f2937;border-radius:10px;padding:12px;border-left:4px solid #27ae60;">
  <div style="color:#6ee7b7;font-size:0.8rem;">RANGE LOW</div>
  <div style="font-size:1.3rem;font-weight:700;color:#fff;">{_fmt_px(state.range_low)}</div>
  <div style="color:#9ca3af;font-size:0.8rem;">突破 {state.low_touches} 次</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"{ok} 有效={state.range_valid} · 手續費OK={state.range_vs_fee_ok} · "
        f"Width={_fmt_px(state.range_width)} · ATR={state.atr:.2f} · "
        f"Break H/L={state.break_count_high}/{state.break_count_low} · 趨勢={state.trend_bias}"
    )

    if state.notes:
        with st.expander("筆記", expanded=False):
            for n in state.notes:
                st.markdown(f"- {n}")


def require_password() -> bool:
    """若設定 APP_PASSWORD（env 或 Streamlit secrets）則要求登入。"""
    password = os.environ.get("APP_PASSWORD", "").strip()
    if not password:
        try:
            password = str(st.secrets.get("APP_PASSWORD", "")).strip()
        except Exception:
            password = ""
    if not password:
        return True
    if st.session_state.get("authed"):
        return True

    st.title("VWAP Range Trader")
    st.caption("請輸入共用密碼（同 fd 用同一個）")
    pw = st.text_input("密碼", type="password")
    if st.button("進入", type="primary"):
        if pw == password:
            st.session_state.authed = True
            st.rerun()
        st.error("密碼錯誤")
    return False


def _display_name() -> str:
    if "client_name" not in st.session_state:
        raw = f"{time.time()}-{os.getpid()}-{id(st.session_state)}"
        st.session_state.client_name = "user-" + hashlib.sha1(raw.encode()).hexdigest()[:6]
    return st.session_state.client_name


def sync_params_into_widgets(shared: dict, base_cfg: dict) -> None:
    """若共用版本較新，覆寫 widget session_state。"""
    version = int(shared["version"])
    params = shared["params"] or default_params(base_cfg)
    local_ver = int(st.session_state.get("_shared_ver", -1))
    if local_ver == version and all(f"p_{k}" in st.session_state for k in PARAM_KEYS):
        return
    for k, v in params.items():
        st.session_state[f"p_{k}"] = v
    st.session_state["_shared_ver"] = version
    st.session_state["_shared_meta"] = {
        "updated_by": shared.get("updated_by", ""),
        "updated_at": shared.get("updated_at", 0),
    }


def collect_params_from_widgets() -> dict:
    return {k: st.session_state[f"p_{k}"] for k in PARAM_KEYS}


st.set_page_config(
    page_title="VWAP Range Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not require_password():
    st.stop()

base_cfg = load_default_cfg()
who = _display_name()

# 先讀共用狀態，同步入 widget keys（必須在 widget 建立前）
shared = load_shared(base_cfg)
sync_params_into_widgets(shared, base_cfg)

# 刷新間隔：Live Track 開住時跟 live_seconds；否則跟參數同步
sync_every = int(os.environ.get("PARAM_SYNC_SECONDS", "3"))
_live_on = bool(st.session_state.get("p_live_track", True))
_live_sec = int(st.session_state.get("p_live_seconds", 3))
poll_sec = max(2, _live_sec if _live_on else sync_every)
try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=poll_sec * 1000, key="param_sync_refresh")
except ImportError:
    st.markdown(
        f"<meta http-equiv='refresh' content='{poll_sec}'>",
        unsafe_allow_html=True,
    )

st.title("VWAP Range Trader")
st.caption("5m candle · 24h rolling VWAP · Live Track · Backtest · 參數即時共用同步")

tab_live, tab_bt = st.tabs(["即時訊號", "Backtest"])

with st.sidebar:
    st.header("控制面板")
    meta = st.session_state.get("_shared_meta", {})
    st.caption(
        f"共用同步 · v{st.session_state.get('_shared_ver', '?')} · "
        f"你是 `{who}` · 上次由 `{meta.get('updated_by') or '—'}`"
    )
    param_sync = st.toggle("啟用參數自動同步", value=True, key="param_sync_on")

    st.toggle("啟用策略評估", key="p_enabled")
    st.toggle("自動刷新策略/K線", key="p_auto_refresh")
    st.slider("策略刷新秒數", 10, 120, step=5, key="p_refresh_seconds")

    st.subheader("Live Track")
    st.toggle("即時追蹤報價 (BTC/USD)", key="p_live_track")
    st.slider("Live 刷新秒數", 2, 15, step=1, key="p_live_seconds")

    st.subheader("市場")
    # selectbox 需要合法 index；用 session 值
    if st.session_state.get("p_symbol") not in SYMBOLS:
        st.session_state["p_symbol"] = SYMBOLS[0]
    st.selectbox(
        "交易對",
        SYMBOLS,
        format_func=lambda s: f"{SYMBOL_LABELS.get(s, s)} ({s})",
        key="p_symbol",
    )
    st.slider("資料回看（小時）", 24, 72, step=6, key="p_lookback_hours")
    st.slider("VWAP Rolling（小時）", 12, 48, step=4, key="p_vwap_hours")

    st.subheader("倉位（相對 VWAP 距離 %）")
    st.number_input("|距離| 近", 0.01, 2.0, step=0.01, key="p_size_near_pct")
    st.number_input("|距離| 中", 0.05, 3.0, step=0.01, key="p_size_mid_pct")
    st.number_input("|距離| 遠", 0.1, 5.0, step=0.05, key="p_size_far_pct")
    st.number_input("基礎名義倉位 (USDT)", 10.0, 100000.0, step=50.0, key="p_base_notional")

    st.subheader("VWAP 交叉 → Continuation")
    st.number_input("4h 交叉上限", 1, 30, step=1, key="p_max_intercept_4h")
    st.number_input("6h 交叉上限", 1, 40, step=1, key="p_max_intercept_6h")

    st.subheader("Range")
    st.number_input("線位容差 %", 0.01, 1.0, step=0.01, key="p_touch_tol")
    st.number_input("同一線位最少突破次數", 2, 20, step=1, key="p_min_touches")
    st.number_input("Entry buffer（range%）", 0.01, 0.3, step=0.01, key="p_entry_buffer_pct")
    st.number_input("Range ≥ Nx 手續費", 1.0, 20.0, step=0.5, key="p_min_range_fee_mult")
    st.number_input("單邊手續費率", 0.0001, 0.002, step=0.0001, format="%.4f", key="p_fee_rate")
    st.number_input("ATR 週期", 5, 50, step=1, key="p_atr_period")
    st.number_input("SL ATR 倍數", 0.5, 3.0, step=0.05, key="p_sl_atr_mult")

    st.subheader("成交量")
    st.number_input("低量門檻 (BTC/5m)", 1.0, 200.0, step=1.0, key="p_min_btc_vol")
    st.number_input("Volume Avg 週期", 5, 60, step=1, key="p_vol_avg_period")
    st.number_input("連續低量次數 → 停 Range", 1, 20, step=1, key="p_low_vol_consec")
    st.number_input("Breakout 量 < Nx Avg", 1.0, 20.0, step=0.5, key="p_breakout_vol_mult")
    st.toggle("啟用 Trailing SL（LH/LL）", key="p_trailing_enabled")

    col_a, col_b = st.columns(2)
    run_btn = col_a.button("重新評估", type="primary", use_container_width=True)
    if col_b.button("重設預設", use_container_width=True):
        save_shared(default_params(base_cfg), base_cfg, updated_by=who)
        st.session_state.pop("_shared_ver", None)
        st.rerun()

# 收集並寫回共用（若有變更）
ui = collect_params_from_widgets()
if st.session_state.get("param_sync_on", True):
    latest = load_shared(base_cfg)
    # 若檔案被對方更新得更新，優先採用對方（避免覆蓋）
    if int(latest["version"]) > int(st.session_state.get("_shared_ver", 0)):
        sync_params_into_widgets(latest, base_cfg)
        ui = collect_params_from_widgets()
        st.rerun()
    else:
        saved = save_shared(ui, base_cfg, updated_by=who)
        st.session_state["_shared_ver"] = saved["version"]
        st.session_state["_shared_meta"] = {
            "updated_by": saved.get("updated_by", ""),
            "updated_at": saved.get("updated_at", 0),
        }

cfg = merge_ui_cfg(base_cfg, ui)
symbol = ui["symbol"]
lookback_hours = ui["lookback_hours"]
base_notional = ui["base_notional"]
enabled = ui["enabled"]
auto_refresh = ui["auto_refresh"]
refresh_seconds = ui["refresh_seconds"]
live_track = ui["live_track"]
live_seconds = ui["live_seconds"]
pair_label = SYMBOL_LABELS.get(symbol, symbol)

if "engine" not in st.session_state:
    st.session_state.engine = StrategyEngine(cfg)
else:
    st.session_state.engine.cfg = cfg

if live_track:
    st.sidebar.caption(f"Live Track 每 {live_seconds}s 更新報價")
if auto_refresh:
    st.sidebar.caption(f"策略/K線 cache ~{min(25, refresh_seconds)}s")


@st.cache_data(ttl=20, show_spinner=False)
def _cached_klines(sym: str, hours: int):
    return fetch_klines_hours(symbol=sym, interval="5m", hours=int(hours))


with tab_live:
    # —— Live Track 報價條 ——
    if live_track:
        try:
            ticker = fetch_ticker_24h(symbol)
            live_price = ticker["price"]
            chg = ticker["change_pct"]
            st.markdown(
                f"""
<div style="
  background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
  border: 1px solid #3d5a6c; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
">
  <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:8px;">
    <div>
      <span style="color:#9fb3c8; font-size:0.85rem;">LIVE · {pair_label}</span>
      <div style="font-size:2rem; font-weight:700; color:#fff; letter-spacing:0.5px;">
        ${live_price:,.2f}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.25rem; font-weight:600; color:{'#2ecc71' if chg >= 0 else '#e74c3c'};">
        {'+' if chg >= 0 else ''}{chg:.2f}%
      </div>
      <div style="color:#9fb3c8; font-size:0.8rem;">
        24h H {ticker['high']:,.0f} · L {ticker['low']:,.0f} · Vol {ticker['volume']:,.0f} BTC
      </div>
      <div style="color:#6c879a; font-size:0.75rem;">
        更新 {ticker['asof'].strftime('%H:%M:%S')} UTC · 來源 {ticker.get('source') or 'market'} · {symbol}
      </div>
    </div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            if "last_vwap" in st.session_state and st.session_state.last_vwap:
                dist = (live_price - st.session_state.last_vwap) / st.session_state.last_vwap * 100.0
                st.caption(f"現價 vs 24h VWAP：{dist:+.3f}%（VWAP {st.session_state.last_vwap:,.2f}）")
        except Exception as e:
            st.warning(f"Live Track 暫時拎唔到報價：{e}")

    if not enabled:
        st.warning("策略評估已關閉。打開側邊欄「啟用策略評估」後再開始。")
    else:
        try:
            with st.spinner(f"抓取 {symbol} 5m K 線…"):
                raw = _cached_klines(symbol, int(lookback_hours))
            closed = drop_incomplete_candle(raw, interval_minutes=5)
            if closed is None or len(closed) < 30:
                st.error("已收線 5m K 線不足，請稍後再試。")
            else:
                df = compute_indicators(closed, cfg)
                state = st.session_state.engine.evaluate(df)
                st.session_state.last_vwap = state.vwap

                sig = state.signal
                suggested_size = base_notional * float(sig.size_mult or state.size_mult)

                now_utc = pd.Timestamp.now(tz="UTC")
                last_closed_open = df.index[-1]
                if last_closed_open.tzinfo is None:
                    last_closed_open = last_closed_open.tz_localize("UTC")
                forming_close_at = last_closed_open + pd.Timedelta(minutes=10)
                secs_left = max(0, int((forming_close_at - now_utc).total_seconds()))
                mm, ss = divmod(secs_left, 60)
                st.caption(
                    f"⏱ 5m candle · 只用已收線 · 上一支 `{last_closed_open.strftime('%H:%M')} UTC` · "
                    f"新 candle 仲有 **{mm:02d}:{ss:02d}**"
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(f"{pair_label}（5m 收線）", f"{state.price:,.2f}")
                c2.metric("24h VWAP", f"{state.vwap:,.2f}", f"{state.vwap_dist_pct:+.3f}%")
                c3.metric("市場模式", state.mode)
                c4.metric("強弱", state.strength)
                c5.metric("建議倉位", f"{suggested_size:,.0f} USDT", f"×{state.size_mult:.2f}")

                c6, c7, c8, c9 = st.columns(4)
                c6.metric("4h VWAP 交叉", state.intercept_4h, "continuation" if state.continuation else "choppy")
                c7.metric("6h VWAP 交叉", state.intercept_6h)
                c8.metric("量 / Avg", f"{state.vol:.1f}", f"avg {state.vol_avg:.1f}")
                c9.metric("低量連續", state.low_vol_streak, "暫停 Range" if state.range_paused else "正常")

                st.divider()
                left, right = st.columns([1.35, 1])
                with left:
                    st.subheader("圖表（24h · 5m 已收線）")
                    st.plotly_chart(build_chart(df, state), use_container_width=True)
                with right:
                    st.subheader("訊號面板")
                    render_signal_panel(state, suggested_size)

                st.divider()
                with st.expander("策略邏輯摘要", expanded=False):
                    st.markdown(
                        """
1. **強弱**：現價對 24h rolling VWAP 嘅距離 % → 控制倉位倍數。
2. **Continuation**：6h 交叉 < 上限 且 4h 交叉 < 上限 → breakout/延續偏向。
3. **Range**：24h 圖入面搵「同一線位突破／穿梭 ≥5 次」做 high/low；range 要 > Nx 手續費。
4. **Range Entry**：buffer = width × entry%；short = high−buffer，long = low+buffer；SL = high/low ± ATR 倍數；TP = 對側 entry。
5. **低量暫停**：5m 量 < 門檻 且低過 avg，連續 N 次 → 停 range，轉等 breakout/reversal。
6. **Breakout/Reversal**：first/second break 且量 < 5×avg → 等下一支；同向量縮跟進，或反向放量做反轉。
7. **無 Range**：改睇單邊升/跌（trend / continuation）。
8. **5m Candle**：策略同圖表只用已收線 K；未滿 5 分鐘嘅 forming bar 唔計。
9. **Backtest**：撳上面「Backtest」分頁，用同一套側邊欄參數跑歷史回測。
                        """
                    )
                src_note = LAST_SOURCE.get("klines") or LAST_SOURCE.get("ticker") or "—"
                st.caption(
                    f"資料截至 {df.index[-1]} · {symbol} · K線來源 {src_note} · 僅訊號參考，非自動下單"
                )
        except Exception as e:
            st.error(f"載入失敗：{e}")

with tab_bt:
    render_backtest_tab(cfg, default_notional=float(base_notional))

_ = run_btn
