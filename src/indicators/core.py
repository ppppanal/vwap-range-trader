"""VWAP / ATR / 成交量等指標計算。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """典型價 * 成交量 嘅 rolling VWAP。"""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    return pv.rolling(window, min_periods=max(1, window // 4)).sum() / df[
        "volume"
    ].rolling(window, min_periods=max(1, window // 4)).sum()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def vwap_distance_pct(price: pd.Series, vwap: pd.Series) -> pd.Series:
    return (price - vwap) / vwap * 100.0


def count_vwap_crosses(close: pd.Series, vwap: pd.Series, lookback: int) -> int:
    """過去 lookback 根內 close 穿越 VWAP 嘅次數。"""
    if lookback <= 1 or len(close) < 2:
        return 0
    seg_c = close.iloc[-lookback:]
    seg_v = vwap.iloc[-lookback:]
    above = seg_c > seg_v
    crossed = above != above.shift(1)
    return int(crossed.iloc[1:].sum())


def drop_incomplete_candle(
    df: pd.DataFrame,
    interval_minutes: int = 5,
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    只保留已收線嘅 5m candle。
    現價所在嗰支未滿 5 分鐘嘅 forming candle 會剔除，
    等 5 分鐘過完先出現新一支。
    """
    if df is None or df.empty:
        return df
    ts_now = now or pd.Timestamp.now(tz="UTC")
    if ts_now.tzinfo is None:
        ts_now = ts_now.tz_localize("UTC")
    else:
        ts_now = ts_now.tz_convert("UTC")

    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    last_open = out.index[-1]
    candle_end = last_open + pd.Timedelta(minutes=interval_minutes)
    if ts_now < candle_end:
        out = out.iloc[:-1]
    return out


def _swing_points(seg: pd.DataFrame, left: int = 2, right: int = 2) -> tuple[list[float], list[float]]:
    """搵局部 swing high / swing low 價位。"""
    highs: list[float] = []
    lows: list[float] = []
    h = seg["high"].values
    l = seg["low"].values
    n = len(seg)
    for i in range(left, n - right):
        window_h = h[i - left : i + right + 1]
        window_l = l[i - left : i + right + 1]
        if h[i] >= window_h.max() and h[i] == window_h.max():
            highs.append(float(h[i]))
        if l[i] <= window_l.min() and l[i] == window_l.min():
            lows.append(float(l[i]))
    return highs, lows


def _count_level_breaks(seg: pd.DataFrame, level: float, tol: float) -> int:
    """
    數「突破／穿梭」同一線位次數。
    定義：前收喺線一邊、今根 high/low 穿過去另一邊（或由下穿上／由上穿下）。
    連續幾根喺同一側唔重複計。
    """
    closes = seg["close"].values
    highs = seg["high"].values
    lows = seg["low"].values
    count = 0
    last_side = 0  # -1 below, +1 above, 0 unknown

    for i in range(len(seg)):
        c = closes[i]
        hi = highs[i]
        lo = lows[i]
        # 今根有冇掂到／穿過呢條帶
        band_lo = level - tol
        band_hi = level + tol
        touched = lo <= band_hi and hi >= band_lo

        side = 0
        if c > level + tol:
            side = 1
        elif c < level - tol:
            side = -1

        crossed = False
        if i > 0:
            prev = closes[i - 1]
            # 收市穿越
            if (prev < level and c > level) or (prev > level and c < level):
                crossed = True
            # wick 穿過去（突破嘗試）
            elif prev <= level and hi >= level + tol * 0.5:
                crossed = True
            elif prev >= level and lo <= level - tol * 0.5:
                crossed = True

        if crossed or (touched and side != 0 and side != last_side and last_side != 0):
            count += 1
            last_side = side if side != 0 else (-last_side if last_side != 0 else last_side)
        elif side != 0:
            last_side = side

    return count


def _cluster_levels(prices: list[float], tol: float) -> list[dict]:
    """將接近嘅 swing 價位聚成線位。"""
    if not prices:
        return []
    prices = sorted(prices)
    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        if abs(p - float(np.mean(clusters[-1]))) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        out.append({"level": float(np.mean(c)), "members": len(c)})
    return out


def find_touch_levels(
    df: pd.DataFrame,
    lookback: int,
    tolerance_pct: float,
    min_touches: int,
) -> dict:
    """
    喺過去 lookback（約 24h）圖入面，搵「同一線位被突破／穿梭 ≥ min_touches 次」嘅位。
    - range_high：符合條件嘅最高線位（阻力）
    - range_low：符合條件嘅最低線位（支撐）
    """
    seg = df.iloc[-lookback:] if len(df) >= lookback else df
    empty = {
        "range_high": None,
        "range_low": None,
        "raw_high": None,
        "raw_low": None,
        "high_touches": 0,
        "low_touches": 0,
        "valid": False,
        "candidate_highs": [],
        "candidate_lows": [],
    }
    if seg.empty or len(seg) < 10:
        return empty

    mid = float(seg["close"].median())
    tol = mid * (tolerance_pct / 100.0)
    # 至少要有少少絕對容差，避免太細
    tol = max(tol, mid * 0.0003)

    swing_h, swing_l = _swing_points(seg, left=2, right=2)
    # 亦加入分位做候選，避免 swing 太少
    q_highs = [float(seg["high"].quantile(q)) for q in (0.85, 0.9, 0.95, 1.0)]
    q_lows = [float(seg["low"].quantile(q)) for q in (0.0, 0.05, 0.1, 0.15)]
    high_clusters = _cluster_levels(swing_h + q_highs, tol)
    low_clusters = _cluster_levels(swing_l + q_lows, tol)

    scored_highs = []
    for cl in high_clusters:
        br = _count_level_breaks(seg, cl["level"], tol)
        scored_highs.append({**cl, "breaks": br})
    scored_lows = []
    for cl in low_clusters:
        br = _count_level_breaks(seg, cl["level"], tol)
        scored_lows.append({**cl, "breaks": br})

    scored_highs.sort(key=lambda x: (x["breaks"] >= min_touches, x["level"]), reverse=True)
    scored_lows.sort(key=lambda x: (x["breaks"] >= min_touches, -x["level"]), reverse=True)

    qual_high = [x for x in scored_highs if x["breaks"] >= min_touches]
    qual_low = [x for x in scored_lows if x["breaks"] >= min_touches]

    # 有效阻力：符合突破次數嘅最高線；支撐：符合嘅最低線
    range_high = max((x["level"] for x in qual_high), default=None)
    range_low = min((x["level"] for x in qual_low), default=None)

    high_touches = 0
    low_touches = 0
    if range_high is not None:
        high_touches = next(x["breaks"] for x in qual_high if x["level"] == range_high)
    if range_low is not None:
        low_touches = next(x["breaks"] for x in qual_low if x["level"] == range_low)

    # 若 high <= low 就無效
    valid = (
        range_high is not None
        and range_low is not None
        and range_high > range_low
    )

    return {
        "range_high": range_high if valid else range_high,
        "range_low": range_low if valid else range_low,
        "raw_high": float(seg["high"].max()),
        "raw_low": float(seg["low"].min()),
        "high_touches": high_touches,
        "low_touches": low_touches,
        "valid": valid,
        "candidate_highs": scored_highs[:5],
        "candidate_lows": scored_lows[:5],
        "tolerance": tol,
    }


def find_aux_breakpoint(
    df: pd.DataFrame,
    *,
    lookback_bars: int = 288,
    pivot_left: int = 2,
    pivot_right: int = 2,
) -> dict:
    """
    1D 輔助線：
    - 起點：absolute high / absolute low（唔係突破線位 range high/low）
    - 終點 breakpoint：喺 abs high~low 價帶／時間窗內，
      「最大成交量」且屬 K 線趨勢反轉（swing high/low）嘅位置
    """
    empty = {
        "valid": False,
        "abs_high": None,
        "abs_high_time": None,
        "abs_low": None,
        "abs_low_time": None,
        "breakpoint_price": None,
        "breakpoint_time": None,
        "breakpoint_volume": None,
        "breakpoint_kind": None,  # swing_high / swing_low
    }
    if df is None or df.empty:
        return empty

    seg = df.iloc[-lookback_bars:] if len(df) >= lookback_bars else df.copy()
    if len(seg) < pivot_left + pivot_right + 3:
        return empty

    abs_high_i = int(seg["high"].values.argmax())
    abs_low_i = int(seg["low"].values.argmin())
    abs_high = float(seg["high"].iloc[abs_high_i])
    abs_low = float(seg["low"].iloc[abs_low_i])
    abs_high_time = seg.index[abs_high_i]
    abs_low_time = seg.index[abs_low_i]

    # 時間窗：abs high 同 abs low 之間（含兩端），若同一根就用成段 1D
    t0 = min(abs_high_i, abs_low_i)
    t1 = max(abs_high_i, abs_low_i)
    if t1 - t0 < 3:
        t0, t1 = 0, len(seg) - 1

    window = seg.iloc[t0 : t1 + 1]
    # 價帶：absolute high/low 之間（range 實體）
    band_hi = abs_high
    band_lo = abs_low

    candidates: list[dict] = []
    h = window["high"].values
    l = window["low"].values
    c = window["close"].values
    v = window["volume"].values
    n = len(window)

    for i in range(pivot_left, n - pivot_right):
        # 價要喺 abs range 內（容許掂到邊界）
        px_hi = float(h[i])
        px_lo = float(l[i])
        if px_hi < band_lo or px_lo > band_hi:
            continue

        is_swing_high = px_hi >= h[i - pivot_left : i + pivot_right + 1].max()
        is_swing_low = px_lo <= l[i - pivot_left : i + pivot_right + 1].min()
        if not (is_swing_high or is_swing_low):
            continue

        # 趨勢反轉確認：swing 前後收市方向改變
        prev_dir = 0
        next_dir = 0
        if i > 0:
            prev_dir = 1 if c[i - 1] > c[max(0, i - 3)] else (-1 if c[i - 1] < c[max(0, i - 3)] else 0)
        if i < n - 1:
            next_dir = 1 if c[min(n - 1, i + 2)] > c[i] else (-1 if c[min(n - 1, i + 2)] < c[i] else 0)

        reversal = False
        kind = None
        if is_swing_high and (prev_dir >= 0 and next_dir <= 0):
            reversal = True
            kind = "swing_high"
        if is_swing_low and (prev_dir <= 0 and next_dir >= 0):
            # 若同時係 high/low，用量決定；否則低位反轉
            if not reversal or float(v[i]) >= (candidates[-1]["volume"] if candidates else 0):
                reversal = True
                kind = "swing_low"

        if not reversal:
            continue

        price = px_hi if kind == "swing_high" else px_lo
        candidates.append(
            {
                "price": float(price),
                "time": window.index[i],
                "volume": float(v[i]),
                "kind": kind,
            }
        )

    if not candidates:
        # fallback：窗內最大量 K，用 close 做 breakpoint
        j = int(window["volume"].values.argmax())
        candidates.append(
            {
                "price": float(window["close"].iloc[j]),
                "time": window.index[j],
                "volume": float(window["volume"].iloc[j]),
                "kind": "max_volume",
            }
        )

    bp = max(candidates, key=lambda x: x["volume"])

    return {
        "valid": True,
        "abs_high": abs_high,
        "abs_high_time": abs_high_time,
        "abs_low": abs_low,
        "abs_low_time": abs_low_time,
        "breakpoint_price": bp["price"],
        "breakpoint_time": bp["time"],
        "breakpoint_volume": bp["volume"],
        "breakpoint_kind": bp["kind"],
        "lookback_bars": len(seg),
    }


def detect_trend_bias(df: pd.DataFrame, vwap: pd.Series, lookback: int = 48) -> str:
    """搵唔到 range 時判斷單邊升/跌。"""
    seg = df.iloc[-lookback:]
    if len(seg) < 10:
        return "neutral"
    net = float(seg["close"].iloc[-1] - seg["close"].iloc[0])
    above_vwap = float((seg["close"] > vwap.loc[seg.index]).mean())
    higher_highs = float(seg["high"].iloc[-1] > seg["high"].iloc[: len(seg) // 2].max())
    lower_lows = float(seg["low"].iloc[-1] < seg["low"].iloc[: len(seg) // 2].min())

    if net > 0 and above_vwap >= 0.6:
        return "bullish_trend"
    if net < 0 and above_vwap <= 0.4:
        return "bearish_trend"
    if higher_highs and net > 0:
        return "bullish_trend"
    if lower_lows and net < 0:
        return "bearish_trend"
    return "neutral"


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    vwap_bars = int(cfg["vwap"]["rolling_hours"] * 60 / 5)
    atr_period = int(cfg["range"]["atr_period"])
    vol_period = int(cfg["volume"]["avg_period"])

    out["vwap"] = rolling_vwap(out, vwap_bars)
    out["atr"] = atr(out, atr_period)
    out["vol_avg"] = out["volume"].rolling(vol_period, min_periods=1).mean()
    out["vwap_dist_pct"] = vwap_distance_pct(out["close"], out["vwap"])
    out["typical"] = (out["high"] + out["low"] + out["close"]) / 3.0
    return out
