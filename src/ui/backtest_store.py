"""Backtest 結果持久化（畀 Web UI 重開睇返）。"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from src.strategy.backtest import BacktestResult

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = Path(
    os.environ.get("BACKTEST_HISTORY_PATH", str(ROOT / "data" / "backtest_history.json"))
)
MAX_RUNS = int(os.environ.get("BACKTEST_HISTORY_MAX", "40"))
_LOCK = threading.Lock()


def _empty() -> dict[str, Any]:
    return {"runs": []}


def _load_raw() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return _empty()
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "runs" not in data:
            return _empty()
        return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def _save_raw(data: dict[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(STORE_PATH)


def result_to_dict(result: BacktestResult, *, meta: Optional[dict] = None) -> dict[str, Any]:
    d = asdict(result)
    # timestamp 轉字串方便 JSON
    for t in d.get("trade_list") or []:
        for k in ("entry_time", "exit_time"):
            if k in t and t[k] is not None:
                t[k] = str(t[k])
    return {
        "id": uuid.uuid4().hex[:12],
        "saved_at": time.time(),
        "saved_at_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "meta": meta or {},
        "result": d,
    }


def dict_to_result(payload: dict[str, Any]) -> BacktestResult:
    r = payload["result"] if "result" in payload else payload
    return BacktestResult(
        label=r.get("label", "saved"),
        symbol=r.get("symbol", "BTCUSDT"),
        bars=int(r.get("bars", 0)),
        start=str(r.get("start", "")),
        end=str(r.get("end", "")),
        trades=int(r.get("trades", 0)),
        wins=int(r.get("wins", 0)),
        losses=int(r.get("losses", 0)),
        win_rate=float(r.get("win_rate", 0)),
        gross_pnl_usdt=float(r.get("gross_pnl_usdt", 0)),
        net_pnl_usdt=float(r.get("net_pnl_usdt", 0)),
        fees_usdt=float(r.get("fees_usdt", 0)),
        max_drawdown_usdt=float(r.get("max_drawdown_usdt", 0)),
        avg_pnl_pct=float(r.get("avg_pnl_pct", 0)),
        by_type=r.get("by_type") or {},
        trade_list=r.get("trade_list") or [],
    )


def save_run(result: BacktestResult, *, meta: Optional[dict] = None) -> dict[str, Any]:
    with _LOCK:
        data = _load_raw()
        entry = result_to_dict(result, meta=meta)
        data["runs"].insert(0, entry)
        data["runs"] = data["runs"][:MAX_RUNS]
        _save_raw(data)
        return entry


def list_runs() -> list[dict[str, Any]]:
    with _LOCK:
        data = _load_raw()
    out = []
    for r in data.get("runs") or []:
        res = r.get("result") or {}
        out.append(
            {
                "id": r.get("id"),
                "saved_at_iso": r.get("saved_at_iso"),
                "label": res.get("label"),
                "symbol": res.get("symbol"),
                "trades": res.get("trades"),
                "win_rate": res.get("win_rate"),
                "net_pnl_usdt": res.get("net_pnl_usdt"),
                "start": res.get("start"),
                "end": res.get("end"),
                "meta": r.get("meta") or {},
            }
        )
    return out


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        data = _load_raw()
    for r in data.get("runs") or []:
        if r.get("id") == run_id:
            return r
    return None


def delete_run(run_id: str) -> bool:
    with _LOCK:
        data = _load_raw()
        before = len(data["runs"])
        data["runs"] = [r for r in data["runs"] if r.get("id") != run_id]
        if len(data["runs"]) == before:
            return False
        _save_raw(data)
        return True
