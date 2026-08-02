"""逐根 5m K 線回測策略引擎。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators.core import compute_indicators
from src.strategy.engine import StrategyEngine


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    trade_type: str
    entry: float
    exit: float
    stop_loss: float
    take_profit: float
    size_mult: float
    pnl_pct: float
    pnl_usdt: float
    reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    label: str
    symbol: str
    bars: int
    start: str
    end: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl_usdt: float
    net_pnl_usdt: float
    fees_usdt: float
    max_drawdown_usdt: float
    avg_pnl_pct: float
    by_type: dict
    trade_list: list


def _round_trip_fee_pct(fee_rate: float) -> float:
    return fee_rate * 2.0 * 100.0  # % of notional


def run_backtest(
    raw: pd.DataFrame,
    cfg: dict,
    *,
    label: str = "run",
    base_notional: float = 1000.0,
    warmup_bars: int = 300,
    max_hold_bars: int = 72,
) -> BacktestResult:
    """
    用完整 raw OHLCV 做指標，再由 warmup 開始逐根評估並模擬倉位。
    同一時間最多一倉；觸及 SL/TP 或逾時平倉。
    """
    df = compute_indicators(raw, cfg)
    engine = StrategyEngine(cfg)
    fee_rate = float(cfg["range"]["fee_rate"])
    fee_pct = _round_trip_fee_pct(fee_rate)

    trades: list[Trade] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    open_side: Optional[str] = None
    open_entry = 0.0
    open_sl = 0.0
    open_tp = 0.0
    open_size = 1.0
    open_type = ""
    open_reason = ""
    open_time: Optional[pd.Timestamp] = None
    open_i = 0
    trailing: Optional[float] = None

    start_i = max(warmup_bars, 50)
    if len(df) <= start_i + 5:
        return BacktestResult(
            label=label,
            symbol=cfg.get("symbol", "BTCUSDT"),
            bars=len(df),
            start=str(df.index[0]),
            end=str(df.index[-1]),
            trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            gross_pnl_usdt=0.0,
            net_pnl_usdt=0.0,
            fees_usdt=0.0,
            max_drawdown_usdt=0.0,
            avg_pnl_pct=0.0,
            by_type={},
            trade_list=[],
        )

    for i in range(start_i, len(df)):
        window = df.iloc[: i + 1]
        bar = df.iloc[i]
        ts = df.index[i]

        # 管理持倉：用當根 high/low 檢查 SL/TP
        if open_side is not None:
            hit_sl = False
            hit_tp = False
            exit_px = float(bar["close"])
            exit_reason = "time"

            sl = trailing if (trailing is not None and open_side == "short") else open_sl

            if open_side == "long":
                if float(bar["low"]) <= sl:
                    hit_sl = True
                    exit_px = sl
                    exit_reason = "sl"
                elif float(bar["high"]) >= open_tp:
                    hit_tp = True
                    exit_px = open_tp
                    exit_reason = "tp"
            else:  # short
                if float(bar["high"]) >= sl:
                    hit_sl = True
                    exit_px = sl
                    exit_reason = "sl"
                elif float(bar["low"]) <= open_tp:
                    hit_tp = True
                    exit_px = open_tp
                    exit_reason = "tp"

            held = i - open_i
            if hit_sl or hit_tp or held >= max_hold_bars:
                if open_side == "long":
                    pnl_pct = (exit_px - open_entry) / open_entry * 100.0
                else:
                    pnl_pct = (open_entry - exit_px) / open_entry * 100.0
                notional = base_notional * open_size
                gross = notional * (pnl_pct / 100.0)
                fee = notional * (fee_pct / 100.0)
                net = gross - fee
                equity += net
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)

                trades.append(
                    Trade(
                        entry_time=open_time,
                        exit_time=ts,
                        side=open_side,
                        trade_type=open_type,
                        entry=open_entry,
                        exit=exit_px,
                        stop_loss=sl,
                        take_profit=open_tp,
                        size_mult=open_size,
                        pnl_pct=pnl_pct,
                        pnl_usdt=net,
                        reason=open_reason,
                        exit_reason=exit_reason if (hit_sl or hit_tp) else "max_hold",
                    )
                )
                open_side = None
                trailing = None
                continue

            # 簡化 trailing：持倉中若出現更低高點，收緊 short SL
            if (
                open_side == "short"
                and cfg["breakout"].get("trailing_enabled", True)
                and i >= 3
            ):
                hh = df["high"].iloc[i - 2 : i + 1]
                ll = df["low"].iloc[i - 2 : i + 1]
                if float(hh.iloc[-1]) < float(hh.iloc[0]) and float(ll.iloc[-1]) < float(
                    ll.iloc[0]
                ):
                    cand = float(hh.iloc[-1]) + 1.05 * float(bar["atr"])
                    if trailing is None or cand < trailing:
                        trailing = cand
            continue

        # 無倉 → 評估訊號
        state = engine.evaluate(window)
        sig = state.signal
        if sig.side in ("long", "short") and sig.entry is not None and sig.stop_loss is not None and sig.take_profit is not None:
            # 當根或下一根接近 entry 先當成交（簡化：用 signal entry，若當根觸及）
            entry = float(sig.entry)
            touched = False
            if sig.side == "long" and float(bar["low"]) <= entry <= float(bar["high"]):
                touched = True
            if sig.side == "short" and float(bar["low"]) <= entry <= float(bar["high"]):
                touched = True
            # trend / breakout / reversal：市價約等於 close
            if sig.trade_type in ("trend", "breakout", "reversal"):
                entry = float(bar["close"])
                touched = True

            if touched:
                open_side = sig.side
                open_entry = entry
                open_sl = float(sig.stop_loss)
                open_tp = float(sig.take_profit)
                open_size = float(sig.size_mult or state.size_mult or 1.0)
                open_type = sig.trade_type or state.mode
                open_reason = sig.reason
                open_time = ts
                open_i = i
                trailing = sig.trailing_sl

    # 統計
    wins = sum(1 for t in trades if t.pnl_usdt > 0)
    losses = sum(1 for t in trades if t.pnl_usdt <= 0)
    n = len(trades)
    by_type: dict = {}
    for t in trades:
        by_type.setdefault(t.trade_type or "unknown", {"n": 0, "pnl": 0.0})
        by_type[t.trade_type or "unknown"]["n"] += 1
        by_type[t.trade_type or "unknown"]["pnl"] += t.pnl_usdt

    fees = sum(base_notional * t.size_mult * (fee_pct / 100.0) for t in trades)
    gross = sum(t.pnl_usdt for t in trades) + fees

    return BacktestResult(
        label=label,
        symbol=cfg.get("symbol", "BTCUSDT"),
        bars=len(df),
        start=str(df.index[start_i]),
        end=str(df.index[-1]),
        trades=n,
        wins=wins,
        losses=losses,
        win_rate=(wins / n * 100.0) if n else 0.0,
        gross_pnl_usdt=gross,
        net_pnl_usdt=sum(t.pnl_usdt for t in trades),
        fees_usdt=fees,
        max_drawdown_usdt=max_dd,
        avg_pnl_pct=(float(np.mean([t.pnl_pct for t in trades])) if n else 0.0),
        by_type=by_type,
        trade_list=[asdict(t) for t in trades],
    )


def print_result(r: BacktestResult) -> None:
    print("=" * 60)
    print(f"[{r.label}] {r.symbol}  {r.start} → {r.end}  bars={r.bars}")
    print(
        f"trades={r.trades}  win={r.wins} loss={r.losses}  "
        f"win_rate={r.win_rate:.1f}%  avg_pnl%={r.avg_pnl_pct:.3f}"
    )
    print(
        f"gross={r.gross_pnl_usdt:.2f}  fees={r.fees_usdt:.2f}  "
        f"net={r.net_pnl_usdt:.2f} USDT  maxDD={r.max_drawdown_usdt:.2f}"
    )
    if r.by_type:
        parts = [f"{k}: n={v['n']} pnl={v['pnl']:.2f}" for k, v in r.by_type.items()]
        print("by_type:", " | ".join(parts))
    print("=" * 60)
