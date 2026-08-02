"""跨使用者共用側邊欄參數（檔案 + process 記憶體）。"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SHARED_PATH = Path(os.environ.get("SHARED_PARAMS_PATH", str(ROOT / "config" / "shared_params.json")))

_LOCK = threading.Lock()
_MEM: dict[str, Any] = {"version": 0, "updated_at": 0.0, "updated_by": "", "params": None}

PARAM_KEYS = [
    "enabled",
    "auto_refresh",
    "refresh_seconds",
    "live_track",
    "live_seconds",
    "symbol",
    "lookback_hours",
    "vwap_hours",
    "size_near_pct",
    "size_mid_pct",
    "size_far_pct",
    "base_notional",
    "max_intercept_4h",
    "max_intercept_6h",
    "touch_tol",
    "min_touches",
    "entry_buffer_pct",
    "min_range_fee_mult",
    "fee_rate",
    "atr_period",
    "sl_atr_mult",
    "min_btc_vol",
    "vol_avg_period",
    "low_vol_consec",
    "breakout_vol_mult",
    "trailing_enabled",
]


def default_params(base_cfg: dict) -> dict[str, Any]:
    return {
        "enabled": True,
        "auto_refresh": True,
        "refresh_seconds": int(base_cfg.get("refresh_seconds", 30)),
        "live_track": True,
        "live_seconds": 3,
        "symbol": base_cfg.get("symbol", "BTCUSDT"),
        "lookback_hours": int(base_cfg.get("lookback_hours", 36)),
        "vwap_hours": int(base_cfg["vwap"]["rolling_hours"]),
        "size_near_pct": float(base_cfg["strength"]["size_near_pct"]),
        "size_mid_pct": float(base_cfg["strength"]["size_mid_pct"]),
        "size_far_pct": float(base_cfg["strength"]["size_far_pct"]),
        "base_notional": 1000.0,
        "max_intercept_4h": int(base_cfg["intercept"]["max_4h"]),
        "max_intercept_6h": int(base_cfg["intercept"]["max_6h"]),
        "touch_tol": float(base_cfg["range"]["touch_tolerance_pct"]),
        "min_touches": int(base_cfg["range"]["min_touches"]),
        "entry_buffer_pct": float(base_cfg["range"]["entry_buffer_pct"]),
        "min_range_fee_mult": float(base_cfg["range"]["min_range_fee_mult"]),
        "fee_rate": float(base_cfg["range"]["fee_rate"]),
        "atr_period": int(base_cfg["range"]["atr_period"]),
        "sl_atr_mult": float(base_cfg["range"]["sl_atr_mult"]),
        "min_btc_vol": float(base_cfg["volume"]["min_btc_threshold"]),
        "vol_avg_period": int(base_cfg["volume"]["avg_period"]),
        "low_vol_consec": int(base_cfg["volume"]["low_vol_consecutive"]),
        "breakout_vol_mult": float(base_cfg["volume"]["breakout_max_vol_mult"]),
        "trailing_enabled": bool(base_cfg["breakout"]["trailing_enabled"]),
    }


def _normalize(params: dict[str, Any], base_cfg: dict) -> dict[str, Any]:
    out = default_params(base_cfg)
    for k in PARAM_KEYS:
        if k in params:
            out[k] = params[k]
    # 型別修正
    out["enabled"] = bool(out["enabled"])
    out["auto_refresh"] = bool(out["auto_refresh"])
    out["live_track"] = bool(out["live_track"])
    out["trailing_enabled"] = bool(out["trailing_enabled"])
    out["symbol"] = str(out["symbol"])
    for ik in (
        "refresh_seconds",
        "live_seconds",
        "lookback_hours",
        "vwap_hours",
        "max_intercept_4h",
        "max_intercept_6h",
        "min_touches",
        "atr_period",
        "vol_avg_period",
        "low_vol_consec",
    ):
        out[ik] = int(out[ik])
    out["live_seconds"] = max(2, min(30, int(out["live_seconds"])))
    for fk in (
        "size_near_pct",
        "size_mid_pct",
        "size_far_pct",
        "base_notional",
        "touch_tol",
        "entry_buffer_pct",
        "min_range_fee_mult",
        "fee_rate",
        "sl_atr_mult",
        "min_btc_vol",
        "breakout_vol_mult",
    ):
        out[fk] = float(out[fk])
    return out


def load_shared(base_cfg: dict) -> dict[str, Any]:
    """回傳 {version, updated_at, updated_by, params}。"""
    with _LOCK:
        if SHARED_PATH.exists():
            try:
                data = json.loads(SHARED_PATH.read_text(encoding="utf-8"))
                params = _normalize(data.get("params") or data, base_cfg)
                version = int(data.get("version", 1))
                updated_at = float(data.get("updated_at", SHARED_PATH.stat().st_mtime))
                updated_by = str(data.get("updated_by", ""))
                _MEM.update(
                    {
                        "version": version,
                        "updated_at": updated_at,
                        "updated_by": updated_by,
                        "params": params,
                    }
                )
                return copy.deepcopy(_MEM)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

        if _MEM["params"] is None:
            _MEM["params"] = default_params(base_cfg)
            _MEM["version"] = 1
            _MEM["updated_at"] = time.time()
            _MEM["updated_by"] = "system"
            _atomic_write(_MEM)
        return copy.deepcopy(_MEM)


def save_shared(
    params: dict[str, Any],
    base_cfg: dict,
    *,
    updated_by: str = "user",
) -> dict[str, Any]:
    with _LOCK:
        normalized = _normalize(params, base_cfg)
        prev = _MEM.get("params")
        if prev is not None and prev == normalized:
            return copy.deepcopy(_MEM)

        _MEM["params"] = normalized
        _MEM["version"] = int(_MEM.get("version") or 0) + 1
        _MEM["updated_at"] = time.time()
        _MEM["updated_by"] = updated_by
        _atomic_write(_MEM)
        return copy.deepcopy(_MEM)


def _atomic_write(payload: dict[str, Any]) -> None:
    SHARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SHARED_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "version": payload["version"],
                "updated_at": payload["updated_at"],
                "updated_by": payload["updated_by"],
                "params": payload["params"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(SHARED_PATH)
