"""跑多組 backtest。用法: python run_backtest.py"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from src.data.binance import fetch_klines_hours
from src.strategy.backtest import print_result, run_backtest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "backtest_results.json"


def load_cfg() -> dict:
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    base = load_cfg()
    symbol = base.get("symbol", "BTCUSDT")
    print(f"抓取 {symbol} 5m 資料（約 7 日）…")
    raw = fetch_klines_hours(symbol=symbol, interval="5m", hours=168)
    print(f"取得 {len(raw)} 根 K 線：{raw.index[0]} → {raw.index[-1]}")

    runs = []

    # Run 1: 預設參數，近 7 日
    cfg1 = copy.deepcopy(base)
    cfg1["symbol"] = symbol
    r1 = run_backtest(raw, cfg1, label="default_7d", base_notional=1000.0, warmup_bars=300)
    print_result(r1)
    runs.append(r1)

    # Run 2: 放寬 range 掂次數 / 容差
    cfg2 = copy.deepcopy(base)
    cfg2["symbol"] = symbol
    cfg2["range"]["min_touches"] = 3
    cfg2["range"]["touch_tolerance_pct"] = 0.25
    r2 = run_backtest(raw, cfg2, label="loose_range_7d", base_notional=1000.0, warmup_bars=300)
    print_result(r2)
    runs.append(r2)

    # Run 3: 更易觸發 continuation（提高交叉上限）
    cfg3 = copy.deepcopy(base)
    cfg3["symbol"] = symbol
    cfg3["intercept"]["max_4h"] = 8
    cfg3["intercept"]["max_6h"] = 14
    cfg3["range"]["min_touches"] = 3
    r3 = run_backtest(raw, cfg3, label="loose_continuation_7d", base_notional=1000.0, warmup_bars=300)
    print_result(r3)
    runs.append(r3)

    # Run 4: 只用最近約 3 日窗口
    raw3 = raw.iloc[-(36 * 12 + 50) :]  # ~3d + buffer
    cfg4 = copy.deepcopy(base)
    cfg4["symbol"] = symbol
    cfg4["range"]["min_touches"] = 3
    r4 = run_backtest(raw3, cfg4, label="default_loose_3d", base_notional=1000.0, warmup_bars=280)
    print_result(r4)
    runs.append(r4)

    # Run 5: 較嚴 SL / 較細倉（遠 VWAP 更細）
    cfg5 = copy.deepcopy(base)
    cfg5["symbol"] = symbol
    cfg5["range"]["min_touches"] = 3
    cfg5["range"]["sl_atr_mult"] = 1.5
    cfg5["strength"]["size_far"] = 0.2
    cfg5["strength"]["size_extreme"] = 0.05
    r5 = run_backtest(raw, cfg5, label="wider_sl_7d", base_notional=1000.0, warmup_bars=300)
    print_result(r5)
    runs.append(r5)

    summary = []
    for r in runs:
        summary.append(
            {
                "label": r.label,
                "trades": r.trades,
                "win_rate": round(r.win_rate, 2),
                "net_pnl_usdt": round(r.net_pnl_usdt, 2),
                "max_drawdown_usdt": round(r.max_drawdown_usdt, 2),
                "avg_pnl_pct": round(r.avg_pnl_pct, 4),
                "by_type": r.by_type,
                "start": r.start,
                "end": r.end,
            }
        )

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n摘要已寫入 {OUT}")
    print("\n=== 總覽 ===")
    for s in summary:
        print(
            f"{s['label']:28s}  trades={s['trades']:3d}  "
            f"WR={s['win_rate']:5.1f}%  net={s['net_pnl_usdt']:8.2f}  "
            f"DD={s['max_drawdown_usdt']:7.2f}"
        )


if __name__ == "__main__":
    main()
