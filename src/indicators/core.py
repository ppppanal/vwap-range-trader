"""VWAP / ATR / 成交量等指標計算。"""

from __future__ import annotations

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
    # 第一根無前值，唔計
    return int(crossed.iloc[1:].sum())


def find_touch_levels(
    df: pd.DataFrame,
    lookback: int,
    tolerance_pct: float,
    min_touches: int,
) -> dict:
    """
    喺過去 lookback 根搵「掂過夠多次」嘅高/低水平。
    用最近 N 根嘅最高/最低做候選，再數幾多次掂到容差內。
    """
    seg = df.iloc[-lookback:] if len(df) >= lookback else df
    if seg.empty:
        return {"range_high": None, "range_low": None, "high_touches": 0, "low_touches": 0}

    raw_high = float(seg["high"].max())
    raw_low = float(seg["low"].min())
    tol_h = raw_high * (tolerance_pct / 100.0)
    tol_l = raw_low * (tolerance_pct / 100.0)

    # 燭體 high/low 進入「極值 ± 容差」帶先算掂到一次
    high_touch_mask = seg["high"] >= (raw_high - tol_h)
    low_touch_mask = seg["low"] <= (raw_low + tol_l)

    # 合併連續掂觸為一次（避免同一段橫盤重複計）
    def _distinct_touches(mask: pd.Series) -> int:
        if mask.empty:
            return 0
        prev = False
        count = 0
        for v in mask.astype(bool).tolist():
            if v and not prev:
                count += 1
            prev = v
        return count

    high_touches = _distinct_touches(high_touch_mask)
    low_touches = _distinct_touches(low_touch_mask)

    range_high = raw_high if high_touches >= min_touches else None
    range_low = raw_low if low_touches >= min_touches else None

    return {
        "range_high": range_high,
        "range_low": range_low,
        "raw_high": raw_high,
        "raw_low": raw_low,
        "high_touches": high_touches,
        "low_touches": low_touches,
        "valid": range_high is not None and range_low is not None,
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
