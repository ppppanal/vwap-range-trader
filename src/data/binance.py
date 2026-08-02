"""Binance 公開 K 線資料抓取。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 500,
    end_time_ms: Optional[int] = None,
) -> pd.DataFrame:
    """抓取最多 1000 根 K 線，回傳 OHLCV DataFrame。"""
    params: dict = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def fetch_klines_hours(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    hours: int = 36,
) -> pd.DataFrame:
    """按小時數估算需要嘅根數再抓取。"""
    minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}.get(interval, 5)
    bars = int(hours * 60 / minutes) + 10
    if bars <= 1000:
        return fetch_klines(symbol, interval, limit=bars)

    # 分段抓超過 1000 根
    chunks: list[pd.DataFrame] = []
    end_ms: Optional[int] = None
    remaining = bars
    while remaining > 0:
        batch = min(remaining, 1000)
        part = fetch_klines(symbol, interval, limit=batch, end_time_ms=end_ms)
        if part.empty:
            break
        chunks.append(part)
        end_ms = int(part.index[0].timestamp() * 1000) - 1
        remaining -= len(part)
        if len(part) < batch:
            break

    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep="last")]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
