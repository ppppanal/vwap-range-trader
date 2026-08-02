"""公開行情抓取（多來源 fallback，避開 Streamlit Cloud 上 Binance 451）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# Binance 官方喺部分雲端 IP 會回 451；data-api / 其他所做備援
BINANCE_KLINE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
]
BINANCE_TICKER_URLS = [
    "https://data-api.binance.vision/api/v3/ticker/24hr",
    "https://api.binance.com/api/v3/ticker/24hr",
    "https://api1.binance.com/api/v3/ticker/24hr",
]
BINANCE_PRICE_URLS = [
    "https://data-api.binance.vision/api/v3/ticker/price",
    "https://api.binance.com/api/v3/ticker/price",
]

_HEADERS = {
    "User-Agent": "vwap-range-trader/1.0 (+https://github.com/ppppanal/vwap-range-trader)",
    "Accept": "application/json",
}

# 最近一次成功來源（方便 UI 顯示）
LAST_SOURCE = {"klines": "", "ticker": ""}


def _get_json(url: str, params: dict, timeout: int = 20):
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _parse_binance_klines(raw: list) -> pd.DataFrame:
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
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume", "quote_volume", "trades"]]


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    limit: int,
    end_time_ms: Optional[int],
) -> pd.DataFrame:
    params: dict = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    errors: list[str] = []
    for url in BINANCE_KLINE_URLS:
        try:
            raw = _get_json(url, params)
            if not raw:
                continue
            LAST_SOURCE["klines"] = url.split("/")[2]
            return _parse_binance_klines(raw)
        except Exception as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("Binance klines 全部失敗 → " + " | ".join(errors[-3:]))


def _bybit_interval(interval: str) -> str:
    return {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60"}.get(interval, "5")


def _fetch_bybit_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Bybit spot kline fallback。"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": _bybit_interval(interval),
        "limit": min(limit, 1000),
    }
    data = _get_json(url, params)
    rows = data.get("result", {}).get("list") or []
    if not rows:
        raise RuntimeError("Bybit 無資料")
    # Bybit: [start, open, high, low, close, volume, turnover] 新→舊
    records = []
    for r in rows:
        ts = int(r[0])
        records.append(
            {
                "open_time": pd.to_datetime(ts, unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "quote_volume": float(r[6]),
                "trades": 0,
            }
        )
    df = pd.DataFrame(records).set_index("open_time").sort_index()
    LAST_SOURCE["klines"] = "api.bybit.com"
    return df


def _okx_inst(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    if s.endswith("USD"):
        return f"{s[:-3]}-USD"
    return s


def _okx_bar(interval: str) -> str:
    return {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "1h": "1H"}.get(interval, "5m")


def _fetch_okx_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": _okx_inst(symbol),
        "bar": _okx_bar(interval),
        "limit": str(min(limit, 300)),
    }
    data = _get_json(url, params)
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError("OKX 無資料")
    # OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm] 新→舊
    records = []
    for r in rows:
        records.append(
            {
                "open_time": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "quote_volume": float(r[7]) if len(r) > 7 else float(r[6]),
                "trades": 0,
            }
        )
    df = pd.DataFrame(records).set_index("open_time").sort_index()
    LAST_SOURCE["klines"] = "www.okx.com"
    return df


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 500,
    end_time_ms: Optional[int] = None,
) -> pd.DataFrame:
    """抓取 OHLCV；多來源 fallback。"""
    errors: list[str] = []
    try:
        return _fetch_binance_klines(symbol, interval, limit, end_time_ms)
    except Exception as e:
        errors.append(str(e))

    # 其他所唔支援 endTime 分段時，只喺無 end_time 時用
    if end_time_ms is None:
        for fn in (_fetch_bybit_klines, _fetch_okx_klines):
            try:
                return fn(symbol, interval, limit)
            except Exception as e:
                errors.append(f"{fn.__name__}: {e}")

    raise RuntimeError("所有行情來源失敗：" + " || ".join(errors[-4:]))


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

    # 優先試一次性大 limit（Bybit/OKX/Binance vision）
    try:
        df = fetch_klines(symbol, interval, limit=min(bars, 1000))
        if len(df) >= min(bars, 200):
            return df
    except Exception:
        pass

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


def _ticker_from_binance(symbol: str) -> dict:
    params = {"symbol": symbol.upper()}
    errors: list[str] = []
    for url in BINANCE_TICKER_URLS:
        try:
            d = _get_json(url, params, timeout=10)
            LAST_SOURCE["ticker"] = url.split("/")[2]
            return {
                "symbol": d.get("symbol", symbol.upper()),
                "price": float(d["lastPrice"]),
                "open": float(d["openPrice"]),
                "high": float(d["highPrice"]),
                "low": float(d["lowPrice"]),
                "change_pct": float(d["priceChangePercent"]),
                "change": float(d["priceChange"]),
                "volume": float(d["volume"]),
                "quote_volume": float(d["quoteVolume"]),
                "asof": datetime.now(timezone.utc),
                "source": LAST_SOURCE["ticker"],
            }
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("Binance ticker 失敗：" + " | ".join(errors[-2:]))


def _ticker_from_bybit(symbol: str) -> dict:
    url = "https://api.bybit.com/v5/market/tickers"
    data = _get_json(url, {"category": "spot", "symbol": symbol.upper()}, timeout=10)
    lst = data.get("result", {}).get("list") or []
    if not lst:
        raise RuntimeError("Bybit ticker 無資料")
    d = lst[0]
    last = float(d["lastPrice"])
    prev = float(d.get("prevPrice24h") or last)
    change = last - prev
    change_pct = (change / prev * 100.0) if prev else 0.0
    LAST_SOURCE["ticker"] = "api.bybit.com"
    return {
        "symbol": symbol.upper(),
        "price": last,
        "open": prev,
        "high": float(d.get("highPrice24h") or last),
        "low": float(d.get("lowPrice24h") or last),
        "change_pct": change_pct,
        "change": change,
        "volume": float(d.get("volume24h") or 0),
        "quote_volume": float(d.get("turnover24h") or 0),
        "asof": datetime.now(timezone.utc),
        "source": LAST_SOURCE["ticker"],
    }


def _ticker_from_okx(symbol: str) -> dict:
    url = "https://www.okx.com/api/v5/market/ticker"
    data = _get_json(url, {"instId": _okx_inst(symbol)}, timeout=10)
    lst = data.get("data") or []
    if not lst:
        raise RuntimeError("OKX ticker 無資料")
    d = lst[0]
    last = float(d["last"])
    open_24h = float(d.get("open24h") or last)
    change = last - open_24h
    change_pct = (change / open_24h * 100.0) if open_24h else 0.0
    LAST_SOURCE["ticker"] = "www.okx.com"
    return {
        "symbol": symbol.upper(),
        "price": last,
        "open": open_24h,
        "high": float(d.get("high24h") or last),
        "low": float(d.get("low24h") or last),
        "change_pct": change_pct,
        "change": change,
        "volume": float(d.get("vol24h") or 0),
        "quote_volume": float(d.get("volCcy24h") or 0),
        "asof": datetime.now(timezone.utc),
        "source": LAST_SOURCE["ticker"],
    }


def fetch_ticker_24h(symbol: str = "BTCUSDT") -> dict:
    """即時 24h ticker；多來源 fallback。"""
    errors: list[str] = []
    for fn in (_ticker_from_binance, _ticker_from_bybit, _ticker_from_okx):
        try:
            return fn(symbol)
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("Ticker 全部失敗：" + " || ".join(errors[-3:]))


def fetch_last_price(symbol: str = "BTCUSDT") -> float:
    params = {"symbol": symbol.upper()}
    for url in BINANCE_PRICE_URLS:
        try:
            return float(_get_json(url, params, timeout=8)["price"])
        except Exception:
            continue
    return float(fetch_ticker_24h(symbol)["price"])


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
