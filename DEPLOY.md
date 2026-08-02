# 長期 Host 部署

## 方案 A：Render（建議）

1. 將本專案推上 GitHub
2. 去 [https://render.com](https://render.com) → New → Blueprint → 選呢個 repo（會讀 `render.yaml`）
3. 喺 Environment 設定 `APP_PASSWORD`（你同 fd 共用密碼）
4. Deploy 完會有 `https://xxxx.onrender.com`

Free plan 閒置會休眠，第一次打開可能要等 30–60 秒。

## 方案 B：Streamlit Community Cloud

1. 推上 GitHub
2. 去 [https://share.streamlit.io](https://share.streamlit.io) → New app
3. Main file 填 `app.py`
4. Secrets / 環境變數加：

```toml
APP_PASSWORD = "你嘅密碼"
```

（Streamlit Cloud 用 Secrets；若要用 env，喺進階設定加。）

## 方案 C：Docker（VPS）

```bash
docker build -t vwap-range-trader .
docker run -d -p 8501:8501 \
  -e APP_PASSWORD='你嘅密碼' \
  -v vwapdata:/data \
  --name vwap vwap-range-trader
```

## 參數同步

- 側邊欄開住「啟用參數自動同步」
- 任何一人改參數會寫入 `shared_params.json`
- 其他人約 3 秒內自動對齊（同一 Host 實例）
