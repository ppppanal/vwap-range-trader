# VWAP Range Trader

用 Binance 公開 5 分鐘 K 線 + 24 小時 rolling VWAP，做 Range / Breakout / Reversal 訊號評估，並提供 Streamlit 控制面板。

## 安裝

```bash
cd vwap-range-trader
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 啟動

```bash
streamlit run app.py
```

瀏覽器打開後，用左側控制面板調參數，主畫面會顯示強弱、VWAP 距離、range、訊號同圖表。

## 策略重點

| 模組 | 說明 |
|------|------|
| 強弱 / sizing | 現價 vs 24h VWAP 距離 % 縮放倉位 |
| Continuation | 4h / 6h VWAP 交叉次數低於門檻 |
| Range | 24h 高低位掂夠次數、寬度 > Nx 手續費 |
| 低量暫停 | 連續低於 20 BTC 且低過 avg → 停 range |
| Breakout / Reversal | 突破後睇下一支量縮延續或放量反轉 |
| Trend | 無有效 range 時改睇單邊 |

預設參數喺 `config/default.yaml`，亦可喺 UI 即時改。

## 注意

- 只用公開行情，**唔會自動下單**。
- 訊號僅供研究／手動參考。
