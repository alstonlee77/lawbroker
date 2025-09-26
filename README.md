# lawbroker
# lawbroker（Ollama為主，Gemini為輔）

本專案是保險公司內訓與模擬考網站，支援：
- GitHub 題庫（repo/資料夾/分頁）
- 練習模式（即時判斷）
- 模擬考（交卷後顯示成績、正解、AI 解析、CSV 下載）
- AI 解析：**Ollama 優先**、失敗或未設定時**自動回退 Gemini**

## 1) 題庫來源（GitHub）
在 `secrets` 或環境變數設定：
- `GH_TOKEN`（可選）
- `GH_OWNER`、`GH_REPO`、`GH_BRANCH`（如 main）
- `GH_FOLDER`（例如「題庫」）
並在 `題庫/人身`、`題庫/外幣`、`題庫/投資型` 下放 Excel 題庫。

## 2) LLM 設定（Ollama > Gemini）
### Ollama（建議）
在可連到 Ollama 的環境設定：
- `OLLAMA_ENDPOINT`：例如 `http://127.0.0.1:11434` 或內網 VM
- `OLLAMA_MODEL`：例如 `qwen2.5:7b-instruct-q4_0`、`llama3.1:8b-instruct-q4_0`
- `OLLAMA_TIMEOUT`：預設 120

> 只要設了 `OLLAMA_ENDPOINT` 就會優先用 Ollama。

### Gemini（備援）
若未設 `OLLAMA_ENDPOINT` 或 Ollama 回傳空結果，會回退至 Gemini：
- `GEMINI_API_KEY`
- `GEMINI_MODEL`（預設 `gemini-2.5-flash`，也可 `gemini-1.5-flash-8b`）

## 3) 管理模式
- `ADMIN_PASSWORD`：在側邊欄可啟用管理模式（目前主要保留擴充位；可限制進階操作）

## 4) 啟動
```bash
pip install -r requirements.txt
streamlit run app.py
