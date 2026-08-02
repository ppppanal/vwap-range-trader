"""VWAP Range / Breakout / Reversal 策略引擎。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.indicators.core import (
    count_vwap_crosses,
    detect_trend_bias,
    find_touch_levels,
)


class MarketMode(str, Enum):
    RANGE = "range"
    CONTINUATION = "continuation_breakout"
    BREAKOUT_WATCH = "breakout_watch"
    REVERSAL = "reversal"
    TREND = "trend"
    PAUSED_LOW_VOL = "paused_low_vol"
    IDLE = "idle"


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    side: str = "flat"
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_sl: Optional[float] = None
    size_mult: float = 0.0
    reason: str = ""
    trade_type: str = ""  # range / breakout / reversal / trend


@dataclass
class StrategyState:
    mode: str = MarketMode.IDLE.value
    strength: str = "neutral"  # strong_bull / mild_bull / neutral / mild_bear / strong_bear
    vwap: float = 0.0
    price: float = 0.0
    vwap_dist_pct: float = 0.0
    size_mult: float = 0.0
    intercept_4h: int = 0
    intercept_6h: int = 0
    continuation: bool = False
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_width: Optional[float] = None
    high_touches: int = 0
    low_touches: int = 0
    range_valid: bool = False
    range_vs_fee_ok: bool = False
    atr: float = 0.0
    vol: float = 0.0
    vol_avg: float = 0.0
    low_vol_streak: int = 0
    range_paused: bool = False
    break_count_high: int = 0
    break_count_low: int = 0
    trend_bias: str = "neutral"
    signal: Signal = field(default_factory=Signal)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class StrategyEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._low_vol_streak = 0
        self._break_high_events: list[dict] = []
        self._break_low_events: list[dict] = []
        self._pending_break: Optional[dict] = None
        self._active_signal: Optional[Signal] = None
        self._swing_highs: list[float] = []
        self._swing_lows: list[float] = []

    def reset_runtime(self) -> None:
        self._low_vol_streak = 0
        self._break_high_events.clear()
        self._break_low_events.clear()
        self._pending_break = None
        self._active_signal = None
        self._swing_highs.clear()
        self._swing_lows.clear()

    # ------------------------------------------------------------------ sizing
    def size_from_vwap_distance(self, dist_pct: float) -> float:
        s = self.cfg["strength"]
        ad = abs(dist_pct)
        if ad < s["size_near_pct"]:
            return float(s["size_near"])
        if ad < s["size_mid_pct"]:
            return float(s["size_mid"])
        if ad < s["size_far_pct"]:
            return float(s["size_far"])
        return float(s["size_extreme"])

    def classify_strength(self, dist_pct: float) -> str:
        if dist_pct >= 0.4:
            return "strong_bull"
        if dist_pct >= 0.15:
            return "mild_bull"
        if dist_pct <= -0.4:
            return "strong_bear"
        if dist_pct <= -0.15:
            return "mild_bear"
        return "neutral"

    # ------------------------------------------------------------------ range
    def _range_fee_ok(self, high: float, low: float) -> bool:
        width = high - low
        mid = (high + low) / 2.0
        fee = mid * float(self.cfg["range"]["fee_rate"]) * 2  # round-trip
        return width > fee * float(self.cfg["range"]["min_range_fee_mult"])

    def _range_entries(self, high: float, low: float, atr_val: float) -> dict:
        width = high - low
        buf = width * float(self.cfg["range"]["entry_buffer_pct"])
        sl_mult = float(self.cfg["range"]["sl_atr_mult"])
        short_entry = high - buf
        long_entry = low + buf
        return {
            "short_entry": short_entry,
            "long_entry": long_entry,
            "short_sl": high + sl_mult * atr_val,
            "long_sl": low - sl_mult * atr_val,
            "short_tp": long_entry,  # 對側 entry 做 TP
            "long_tp": short_entry,
            "buffer": buf,
            "width": width,
        }

    # ------------------------------------------------------------------ volume
    def _update_low_vol_streak(self, vol: float, vol_avg: float) -> int:
        thr = float(self.cfg["volume"]["min_btc_threshold"])
        if vol < thr and vol < vol_avg:
            self._low_vol_streak += 1
        else:
            self._low_vol_streak = 0
        return self._low_vol_streak

    # ------------------------------------------------------------------ breaks
    def _detect_breaks(
        self,
        row: pd.Series,
        prev: pd.Series,
        range_high: float,
        range_low: float,
        vol_avg: float,
    ) -> Optional[dict]:
        max_vol = vol_avg * float(self.cfg["volume"]["breakout_max_vol_mult"])
        max_watch = int(self.cfg["breakout"]["max_breaks_to_watch"])

        event = None
        if prev["close"] <= range_high < row["high"] or (
            prev["high"] <= range_high and row["close"] > range_high
        ):
            if row["volume"] < max_vol:
                self._break_high_events.append(
                    {
                        "side": "high",
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "level": range_high,
                    }
                )
                self._break_high_events = self._break_high_events[-max_watch:]
                event = {"type": "break_high", **self._break_high_events[-1]}

        if prev["close"] >= range_low > row["low"] or (
            prev["low"] >= range_low and row["close"] < range_low
        ):
            if row["volume"] < max_vol:
                self._break_low_events.append(
                    {
                        "side": "low",
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "level": range_low,
                    }
                )
                self._break_low_events = self._break_low_events[-max_watch:]
                event = {"type": "break_low", **self._break_low_events[-1]}

        return event

    def _evaluate_follow_through(
        self,
        candle: pd.Series,
        break_evt: dict,
        range_high: float,
        range_low: float,
        atr_val: float,
        vol_avg: float,
        price: float,
    ) -> Optional[Signal]:
        """
        Breakout 後下一支：
        - 同方向但量縮 → continuation（跟 breakout）
        - 反向且量大過 breakout 支 → reversal
        """
        bvol = break_evt["volume"]
        cvol = float(candle["volume"])
        sl_mult = float(self.cfg["range"]["sl_atr_mult"])
        width = range_high - range_low

        if break_evt["type"] == "break_high":
            # 同方向（升）但量縮
            same_dir = float(candle["close"]) > float(candle["open"]) and float(
                candle["close"]
            ) >= break_evt["close"]
            vol_shrink = cvol < bvol
            # 反轉：量大，收返低過 breakout open
            reverse = cvol > bvol and float(candle["close"]) < break_evt["open"]

            if reverse:
                return Signal(
                    side=Side.SHORT.value,
                    entry=price,
                    stop_loss=range_high + sl_mult * atr_val,
                    take_profit=range_low - width,  # range low - range
                    size_mult=self.size_from_vwap_distance(
                        (price - float(candle.get("vwap", price))) / price * 100
                    ),
                    reason="高位假突破後放量反轉",
                    trade_type="reversal",
                )
            if same_dir and vol_shrink:
                return Signal(
                    side=Side.LONG.value,
                    entry=price,
                    stop_loss=range_low - sl_mult * atr_val,
                    take_profit=range_high + width,
                    size_mult=0.5,
                    reason="突破高位後量縮延續",
                    trade_type="breakout",
                )

        if break_evt["type"] == "break_low":
            same_dir = float(candle["close"]) < float(candle["open"]) and float(
                candle["close"]
            ) <= break_evt["close"]
            vol_shrink = cvol < bvol
            reverse = cvol > bvol and float(candle["close"]) > break_evt["open"]

            if reverse:
                return Signal(
                    side=Side.LONG.value,
                    entry=price,
                    stop_loss=range_low - sl_mult * atr_val,
                    take_profit=range_high + width,
                    size_mult=self.size_from_vwap_distance(
                        (price - float(candle.get("vwap", price))) / price * 100
                    ),
                    reason="低位假突破後放量反轉",
                    trade_type="reversal",
                )
            if same_dir and vol_shrink:
                return Signal(
                    side=Side.SHORT.value,
                    entry=price,
                    stop_loss=range_high + sl_mult * atr_val,
                    take_profit=range_low - width,
                    size_mult=0.5,
                    reason="跌破低位後量縮延續",
                    trade_type="breakout",
                )

        # 下一支反向大過 avg volume → 亦可做 TP 參考 / 反轉提示
        if cvol > vol_avg:
            if break_evt["type"] == "break_high" and float(candle["close"]) < float(
                candle["open"]
            ):
                return Signal(
                    side=Side.SHORT.value,
                    entry=price,
                    stop_loss=max(break_evt["high"], range_high) + sl_mult * atr_val,
                    take_profit=range_low,
                    size_mult=0.6,
                    reason="突破後反向放量，傾向反轉",
                    trade_type="reversal",
                )
            if break_evt["type"] == "break_low" and float(candle["close"]) > float(
                candle["open"]
            ):
                return Signal(
                    side=Side.LONG.value,
                    entry=price,
                    stop_loss=min(break_evt["low"], range_low) - sl_mult * atr_val,
                    take_profit=range_high,
                    size_mult=0.6,
                    reason="跌破後反向放量，傾向反轉",
                    trade_type="reversal",
                )
        return None

    def _trailing_for_short(
        self, df: pd.DataFrame, atr_val: float, current_sl: Optional[float]
    ) -> Optional[float]:
        """Breakout short 後若出現 lower high / lower low，收緊 trailing SL。"""
        if len(df) < 6:
            return current_sl
        recent = df.iloc[-6:]
        hh = recent["high"].values
        ll = recent["low"].values
        lower_high = hh[-1] < hh[-3] and hh[-2] < hh[-3]
        lower_low = ll[-1] < ll[-3]
        if lower_high and lower_low:
            candidate = float(hh[-1]) + 1.05 * atr_val
            if current_sl is None or candidate < current_sl:
                return candidate
        return current_sl

    # ------------------------------------------------------------------ main
    def evaluate(self, df: pd.DataFrame) -> StrategyState:
        notes: list[str] = []
        if df is None or len(df) < 30:
            return StrategyState(notes=["資料不足，至少需要約 30 根 5m K 線"])

        row = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(row["close"])
        vwap = float(row["vwap"]) if not np.isnan(row["vwap"]) else price
        dist = float(row["vwap_dist_pct"]) if not np.isnan(row["vwap_dist_pct"]) else 0.0
        atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
        vol = float(row["volume"])
        vol_avg = float(row["vol_avg"]) if not np.isnan(row["vol_avg"]) else vol

        size_mult = self.size_from_vwap_distance(dist)
        strength = self.classify_strength(dist)

        icfg = self.cfg["intercept"]
        intercept_4h = count_vwap_crosses(df["close"], df["vwap"], int(icfg["window_4h_bars"]))
        intercept_6h = count_vwap_crosses(df["close"], df["vwap"], int(icfg["window_6h_bars"]))
        continuation = (
            intercept_6h < int(icfg["max_6h"]) and intercept_4h < int(icfg["max_4h"])
        )
        if continuation:
            notes.append("4h/6h VWAP 交叉少 → continuation / breakout 偏向")

        # range
        bars_24h = int(self.cfg["range"]["lookback_hours"] * 60 / 5)
        levels = find_touch_levels(
            df,
            lookback=bars_24h,
            tolerance_pct=float(self.cfg["range"]["touch_tolerance_pct"]),
            min_touches=int(self.cfg["range"]["min_touches"]),
        )
        range_high = levels["range_high"]
        range_low = levels["range_low"]
        range_valid = bool(levels["valid"])
        range_vs_fee_ok = False
        range_width = None
        entries = None
        if range_valid and range_high is not None and range_low is not None:
            range_width = range_high - range_low
            range_vs_fee_ok = self._range_fee_ok(range_high, range_low)
            if range_vs_fee_ok:
                entries = self._range_entries(range_high, range_low, atr_val)
            else:
                notes.append("Range 細過 5x 手續費，唔做 range trade")
        else:
            notes.append("搵唔到有效 range（24h 內無同一線位突破≥門檻次數）")

        # volume pause
        streak = self._update_low_vol_streak(vol, vol_avg)
        need = int(self.cfg["volume"]["low_vol_consecutive"])
        range_paused = streak >= need
        if range_paused:
            notes.append(f"量連續 {streak} 次 <20BTC 且低過 avg → 暫停 range，等 breakout/reversal")

        mode = MarketMode.IDLE
        signal = Signal(size_mult=size_mult)

        # pending break follow-through（用最新完整 K）
        if self._pending_break is not None and range_high and range_low:
            ft = self._evaluate_follow_through(
                row,
                self._pending_break,
                range_high,
                range_low,
                atr_val,
                vol_avg,
                price,
            )
            if ft is not None:
                ft.size_mult = size_mult if ft.size_mult == 0 else ft.size_mult * size_mult
                if ft.trade_type == "reversal":
                    mode = MarketMode.REVERSAL
                else:
                    mode = MarketMode.BREAKOUT_WATCH
                if ft.side == Side.SHORT.value and self.cfg["breakout"]["trailing_enabled"]:
                    ft.trailing_sl = self._trailing_for_short(df, atr_val, ft.stop_loss)
                signal = ft
                notes.append(ft.reason)
                self._pending_break = None
            else:
                # 未確認，繼續等下一支（只留一輪）
                notes.append("已偵測 break，等待跟進 K 線確認")
                mode = MarketMode.BREAKOUT_WATCH
                self._pending_break = None

        # 偵測新 break（若未產出訊號）
        if signal.side == Side.FLAT.value and range_high and range_low and (
            range_paused or continuation or not range_vs_fee_ok
        ):
            br = self._detect_breaks(row, prev, range_high, range_low, vol_avg)
            if br is not None:
                self._pending_break = br
                mode = MarketMode.BREAKOUT_WATCH
                notes.append(
                    f"偵測到 {br['type']}（量 {br['volume']:.1f} < 5x avg），等下一支確認"
                )

        # Range trade（量未暫停、range 有效）
        if (
            signal.side == Side.FLAT.value
            and range_valid
            and range_vs_fee_ok
            and not range_paused
            and entries is not None
        ):
            mode = MarketMode.RANGE
            # 接近 short / long entry 帶就出訊號
            band = entries["buffer"] * 0.5
            if abs(price - entries["short_entry"]) <= band or price >= entries["short_entry"]:
                if price >= entries["short_entry"] - band:
                    signal = Signal(
                        side=Side.SHORT.value,
                        entry=entries["short_entry"],
                        stop_loss=entries["short_sl"],
                        take_profit=entries["short_tp"],
                        size_mult=size_mult,
                        reason="Range 高位附近 short entry",
                        trade_type="range",
                    )
            if signal.side == Side.FLAT.value and (
                abs(price - entries["long_entry"]) <= band or price <= entries["long_entry"]
            ):
                if price <= entries["long_entry"] + band:
                    signal = Signal(
                        side=Side.LONG.value,
                        entry=entries["long_entry"],
                        stop_loss=entries["long_sl"],
                        take_profit=entries["long_tp"],
                        size_mult=size_mult,
                        reason="Range 低位附近 long entry",
                        trade_type="range",
                    )
            if signal.side == Side.FLAT.value:
                signal = Signal(
                    side=Side.FLAT.value,
                    entry=None,
                    stop_loss=None,
                    take_profit=None,
                    size_mult=size_mult,
                    reason=(
                        f"Range 有效：short@{entries['short_entry']:.1f} / "
                        f"long@{entries['long_entry']:.1f}，等候觸發"
                    ),
                    trade_type="range",
                )
                notes.append(signal.reason)

        # 無有效 range → 單邊
        if signal.side == Side.FLAT.value and (not range_valid or not range_vs_fee_ok):
            bias = detect_trend_bias(df, df["vwap"], lookback=48)
            mode = MarketMode.TREND if bias != "neutral" else (
                MarketMode.CONTINUATION if continuation else MarketMode.IDLE
            )
            sl_mult = float(self.cfg["range"]["sl_atr_mult"])
            if bias == "bullish_trend" or (continuation and dist > 0):
                signal = Signal(
                    side=Side.LONG.value,
                    entry=price,
                    stop_loss=price - sl_mult * atr_val,
                    take_profit=price + 2.0 * atr_val,
                    size_mult=size_mult * (0.7 if continuation else 0.5),
                    reason="無有效 range → 單邊偏多 / continuation long",
                    trade_type="trend",
                )
                notes.append(signal.reason)
            elif bias == "bearish_trend" or (continuation and dist < 0):
                signal = Signal(
                    side=Side.SHORT.value,
                    entry=price,
                    stop_loss=price + sl_mult * atr_val,
                    take_profit=price - 2.0 * atr_val,
                    size_mult=size_mult * (0.7 if continuation else 0.5),
                    reason="無有效 range → 單邊偏空 / continuation short",
                    trade_type="trend",
                )
                notes.append(signal.reason)
            else:
                notes.append("中性，暫無明確單邊")
        else:
            bias = detect_trend_bias(df, df["vwap"], lookback=48)

        if range_paused and mode == MarketMode.RANGE:
            mode = MarketMode.PAUSED_LOW_VOL

        self._active_signal = signal

        return StrategyState(
            mode=mode.value if isinstance(mode, MarketMode) else str(mode),
            strength=strength,
            vwap=vwap,
            price=price,
            vwap_dist_pct=dist,
            size_mult=size_mult,
            intercept_4h=intercept_4h,
            intercept_6h=intercept_6h,
            continuation=continuation,
            range_high=range_high,
            range_low=range_low,
            range_width=range_width,
            high_touches=int(levels["high_touches"]),
            low_touches=int(levels["low_touches"]),
            range_valid=range_valid,
            range_vs_fee_ok=range_vs_fee_ok,
            atr=atr_val,
            vol=vol,
            vol_avg=vol_avg,
            low_vol_streak=streak,
            range_paused=range_paused,
            break_count_high=len(self._break_high_events),
            break_count_low=len(self._break_low_events),
            trend_bias=bias,
            signal=signal,
            notes=notes,
        )
