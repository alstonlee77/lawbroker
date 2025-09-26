# app.py
import os
import io
import re
import time
import json
import random
import datetime as dt
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ======================================
# =========== 基本設定 / 工具 =============
# ======================================
st.set_page_config(
    page_title="模擬考與題庫練習",
    layout="wide",
    page_icon="📘",
)

# --------- util ----------
def _get_secret(k: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(k, default)  # type: ignore[attr-defined]
    except Exception:
        return default

def _qparams() -> Dict[str, str]:
    try:
        return st.query_params.to_dict()
    except Exception:
        # Streamlit < 1.33.0
        try:
            return st.experimental_get_query_params()  # type: ignore[attr-defined]
        except Exception:
            return {}

# --------- GitHub 設定（若用 GitHub 當題庫來源） ----------
GH_TOKEN   = _get_secret("GH_TOKEN",   os.getenv("GH_TOKEN", ""))
GH_OWNER   = _get_secret("GH_OWNER",   os.getenv("GH_OWNER", ""))
GH_REPO    = _get_secret("GH_REPO",    os.getenv("GH_REPO", "lawbroker"))
GH_BRANCH  = _get_secret("GH_BRANCH",  os.getenv("GH_BRANCH", "main"))
GH_FOLDER  = _get_secret("GH_FOLDER",  os.getenv("GH_FOLDER", "題庫"))  # 主資料夾名

# --------- Admin 密碼（管理模式） ----------
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))

# --------- LLM 參數（Ollama 優先，Gemini 後援） ----------
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "gemini")  # 僅在沒設 OLLAMA_ENDPOINT 時會用
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", _get_secret("GEMINI_API_KEY", "")) or ""
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", _get_secret("GEMINI_MODEL", "gemini-2.5-flash")) or "gemini-2.5-flash"

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", _get_secret("OLLAMA_ENDPOINT", ""))  # e.g. http://127.0.0.1:11434
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", _get_secret("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_0")) or "qwen2.5:7b-instruct-q4_0"
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", _get_secret("OLLAMA_TIMEOUT", "120")) or 120)

# ======================================
# ============ 題庫讀取邏輯 ===============
# ======================================

EXT_XLSX = (".xlsx", ".xlsm", ".xls")
EXT_XLS  = (".xls",)

def _is_excel(name: str) -> bool:
    s = name.lower()
    return s.endswith(".xlsx") or s.endswith(".xls") or s.endswith(".xlsm")

def _github_headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        headers["Authorization"] = f"token {GH_TOKEN}"
    return headers

@st.cache_data(show_spinner=False)
def list_github_files_by_domain(domain: str) -> List[str]:
    """
    列出 repo 中 GH_FOLDER/{domain}/ 下所有 Excel 檔案的 download_url。
    """
    if not (GH_OWNER and GH_REPO and GH_BRANCH and GH_FOLDER):
        return []
    api = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{GH_FOLDER}/{domain}?ref={GH_BRANCH}"
    r = requests.get(api, headers=_github_headers(), timeout=30)
    if r.status_code >= 400:
        return []
    data = r.json()
    files = []
    for item in data:
        if item.get("type") == "file" and _is_excel(item.get("name","")):
            files.append(item.get("download_url"))
    return files

@st.cache_data(show_spinner=False)
def fetch_excel_bytes(url: str) -> bytes:
    r = requests.get(url, headers=_github_headers(), timeout=120)
    r.raise_for_status()
    return r.content

def read_one_excel(src: str) -> Dict[str, pd.DataFrame]:
    """
    src 可以是：GitHub raw url、或本機上傳檔（透過 file_uploader 給的名稱/bytes）
    回傳：{sheet_name: df}
    """
    dfs: Dict[str, pd.DataFrame] = {}
    try:
        if src.startswith("http"):
            content = fetch_excel_bytes(src)
            bio = io.BytesIO(content)
        else:
            # 本機路徑（或上傳 bytes 的暫存），這裡假設 src 就是 path-like；如果你要從 file_uploader 直接傳 bytes，請改傳 bytes 再包 BytesIO。
            bio = open(src, "rb")
        try:
            # openpyxl / xlrd 都會自動處理
            x = pd.ExcelFile(bio)
            for s in x.sheet_names:
                try:
                    df = x.parse(s)
                    df["_source_sheet"] = s
                    dfs[s] = df
                except Exception:
                    continue
        finally:
            if hasattr(bio, "close"):
                bio.close()
    except Exception as e:
        st.warning(f"無法讀取：{src}（{e}）")
    return dfs

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    統一欄位名稱為：ID, Question, OptionA..D, Answer, Tag
    支援多種可能名稱：'題目','試題','選項一/二/三/四','OptionA/B/C/D','答案選項1..4','答案','正解','Tag'...
    """
    colmap = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=colmap)

    # 題目欄
    q_col = None
    for cand in ["題目", "試題", "Question", "題幹", "題意"]:
        if cand in df.columns:
            q_col = cand
            break
    if q_col is None:
        raise ValueError("載入題庫失敗：找不到『題目/Question』欄位")

    # 選項
    opt_cols = {}
    # 支援 OptionA..D
    for k, lab in zip(["A","B","C","D"], ["OptionA","OptionB","OptionC","OptionD"]):
        if lab in df.columns:
            opt_cols[k] = lab
    # 支援 選項一/二/三/四
    num_map = {"一":"A","二":"B","三":"C","四":"D"}
    for k_cn, k in num_map.items():
        if f"選項{k_cn}" in df.columns:
            opt_cols[k] = f"選項{k_cn}"
    # 支援 答案選項1..4
    for i,k in enumerate(["A","B","C","D"],1):
        if f"答案選項{i}" in df.columns:
            opt_cols[k] = f"答案選項{i}"
    if len(opt_cols) < 2:
        raise ValueError("載入題庫失敗：題庫至少需要 2 個選項欄 (OptionA/OptionB 或 選項一/二 等)")

    # 答案欄
    ans_col = None
    for cand in ["答案","正解","Answer"]:
        if cand in df.columns:
            ans_col = cand
            break
    if ans_col is None:
        raise ValueError("載入題庫失敗：找不到『答案/正解/Answer』欄位")

    # ID 欄（沒有就自動產生）
    id_col = None
    for cand in ["ID","Id","題號","編號"]:
        if cand in df.columns:
            id_col = cand
            break

    # Tag 欄（沒有就後面用 sheet_name 補）
    tag_col = "Tag" if "Tag" in df.columns else None

    # 建立標準欄位
    out = pd.DataFrame()
    out["ID"] = df[id_col] if id_col else range(1, len(df)+1)
    out["Question"] = df[q_col].astype(str)

    # 填滿 4 個選項（不存在者變為空字串）
    for k in ["A","B","C","D"]:
        src = opt_cols.get(k)
        out[f"Option{k}"] = df[src].astype(str) if src in df.columns else ""

    # Answer：支援「A/B/C/D」或「對應文字」
    raw_ans = df[ans_col].astype(str).str.strip()
    # 若原本是「A/B/C/D」
    if raw_ans.str.match(r"^[ABCD]$", case=False).all():
        out["Answer"] = raw_ans.str.upper()
    else:
        # 與選項逐一比對，找出是哪一個
        def to_letter(x: str, row) -> str:
            for k in ["A","B","C","D"]:
                if x == str(row[f"Option{k}"]):
                    return k
            # 若對不上，先標示空，後續判題時視為錯誤
            return ""
        out["Answer"] = [to_letter(a, r) for a, r in zip(raw_ans, out.to_dict("records"))]

    if tag_col:
        out["Tag"] = df[tag_col].astype(str).fillna("").str.strip()
    else:
        # 若沒 Tag，先空，後續會用 sheet_name 補
        out["Tag"] = ""

    # 來源分頁
    if "_source_sheet" in df.columns:
        out["_source_sheet"] = df["_source_sheet"].astype(str)
    else:
        out["_source_sheet"] = ""

    # 清潔
    out = out.fillna("")
    return out

def assemble_bank(
    domain: str,
    files: List[str],
    chosen_sheets: List[str],
    use_sheet_as_tag: bool,
) -> pd.DataFrame:
    """把多個 Excel 的分頁彙整成一個題庫 df"""
    if not files:
        return pd.DataFrame()

    all_rows = []
    for src in files:
        sheets = read_one_excel(src)
        for name, df in sheets.items():
            if chosen_sheets and name not in chosen_sheets:
                continue
            nd = normalize_columns(df)
            if use_sheet_as_tag:
                nd["Tag"] = nd["Tag"].replace("", name)
            else:
                # 若沒 Tag 且沒打勾 -> 仍用 sheet 補空值
                nd["Tag"] = nd["Tag"].replace("", name)
            nd["__source_file"] = src
            all_rows.append(nd)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    # 基本合理化：Answer 若空則視為無效題
    out = out[out["Answer"].isin(list("ABCD"))].copy()
    out.reset_index(drop=True, inplace=True)
    return out

# ======================================
# ============ LLM 解析邏輯 ==============
# ======================================

def sanitize_gemini_model(m: str) -> str:
    m = (m or "").strip()
    alias = {
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-flash-8b": "gemini-1.5-flash-8b"
    }
    # 盡量原樣，其次提供 fallback
    return alias.get(m, m or "gemini-1.5-flash")

def build_explain_prompt(qrow: pd.Series, your: str) -> str:
    opts = []
    for k in ["A","B","C","D"]:
        if str(qrow[f"Option{k}"]).strip():
            opts.append(f"{k}. {qrow[f'Option{k}']}")
    options_text = "\n".join(opts)

    prompt = (
        "你是保險代理人訓練講師，請用繁體中文、簡潔條列 1–3 點說明：\n"
        "1) 為何正確答案正確\n"
        "2) 為何其他選項不正確\n"
        "3) 若有公式或關鍵字，給最短關鍵提醒\n"
        "不要贅詞與前言，總長不超過 120 字。\n\n"
        f"題目：{qrow['Question']}\n\n"
        f"選項：\n{options_text}\n\n"
        f"考生作答：{your or '未作答'}\n"
        f"正確答案：{qrow['Answer']}\n"
    )
    return prompt

def ollama_explain(prompt: str, model: Optional[str] = None) -> str:
    endpoint = OLLAMA_ENDPOINT or ""
    if not endpoint:
        return "（未設定 OLLAMA_ENDPOINT）"
    mdl = model or OLLAMA_MODEL
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/api/generate",
            json={
                "model": mdl,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        txt = (data.get("response") or "").strip()
        return txt or "（Ollama 暫無輸出）"
    except Exception as e:
        return f"（Ollama 解析失敗：{e}）"

def _extract_gemini_text(resp) -> str:
    # 安全抽取：先 resp.text，失敗再從 candidates.parts 取
    try:
        t = getattr(resp, "text", None)
        if t:
            return t
    except Exception:
        pass
    try:
        if getattr(resp, "candidates", None):
            chunks = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for p in parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            chunks.append(txt)
            if chunks:
                return "\n".join(chunks)
            fr = getattr(resp.candidates[0], "finish_reason", "unknown")
            return f"（AI暫無輸出；finish_reason={fr}）"
    except Exception:
        pass
    return "（AI暫無輸出）"

@st.cache_data(show_spinner=False)
def gemini_explain_cached(prompt: str, model: str) -> str:
    if not GEMINI_API_KEY:
        return "（AI詳解失敗：未設定 GEMINI_API_KEY）"
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        m = sanitize_gemini_model(model)
        g = genai.GenerativeModel(m, system_instruction="請用繁體中文、最多120字，條列1–3點說明，避免贅詞。")
        last_err = ""
        # 兩次嘗試 + 兩組 max_tokens
        for cfg in (256, 384):
            for _ in range(2):
                try:
                    resp = g.generate_content(prompt, generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": cfg,
                        "candidate_count": 1
                    })
                    text = _extract_gemini_text(resp).strip()
                    if text and "AI暫無輸出" not in text:
                        return text
                except Exception as e:
                    last_err = str(e)
                time.sleep(0.6)
        return f"（AI詳解暫無輸出；{last_err or '請稍後重試'}）"
    except Exception as e:
        return f"（AI詳解失敗：{e}）"

def llm_explain(prompt: str) -> str:
    """
    以 Ollama 為主；未設定或錯誤時回退 Gemini。
    """
    if OLLAMA_ENDPOINT:
        text = ollama_explain(prompt)
        # 若 Ollama 回來的字串是錯誤訊息/暫無輸出，再回退 Gemini
        if text.startswith("（Ollama 解析失敗") or "暫無輸出" in text:
            # fall back
            return gemini_explain_cached(prompt, GEMINI_MODEL)
        return text
    else:
        return gemini_explain_cached(prompt, GEMINI_MODEL)

# ======================================
# ============== UI 狀態 ================
# ======================================

def init_state():
    ss = st.session_state
    ss.setdefault("started", False)          # 是否已開始出題
    ss.setdefault("mode", "練習")            # 練習 / 模擬
    ss.setdefault("domain", "")              # 人身/外幣/投資型
    ss.setdefault("files", [])               # 已選檔案（URL或路徑）
    ss.setdefault("sheets", [])              # 已選分頁
    ss.setdefault("use_sheet_as_tag", True)  # 是否用分頁名當 Tag
    ss.setdefault("df_bank", pd.DataFrame())
    ss.setdefault("q_indices", [])           # 題目順序（索引 list）
    ss.setdefault("current_idx", 0)          # 當前題目在 q_indices 中的位置
    ss.setdefault("answers", {})             # {row_index: "A"/"B"/...}
    ss.setdefault("results", {})             # {row_index: bool}
    ss.setdefault("start_time", None)
    ss.setdefault("selected_tags", [])       # 章節/標籤 篩選
    ss.setdefault("question_count", 10)
    ss.setdefault("shuffle", True)
    ss.setdefault("admin_mode", False)

init_state()

# ======================================
# =============== 版面 ================
# ======================================

st.markdown("## 📘 模擬考與題庫練習")

with st.sidebar:
    st.subheader("資料來源與管理")

    src_mode = st.radio("來源模式", ["GitHub / 題庫", "本機檔案"], horizontal=True)

    # 管理模式
    if st.button("啟用管理模式"):
        if not ADMIN_PASSWORD:
            st.session_state.admin_mode = True
            st.success("已啟用管理模式（未設定密碼）")
        else:
            pw = st.text_input("請輸入管理密碼", type="password")
            if pw:
                if pw == ADMIN_PASSWORD:
                    st.session_state.admin_mode = True
                    st.success("已啟用管理模式")
                else:
                    st.error("密碼錯誤")

    st.write("---")
    st.markdown("### 領域選擇")
    # 不預設：顯示提示
    domain = st.selectbox("選擇領域", ["（請選擇）", "人身", "外幣", "投資型"], index=0, key="domain")
    if domain == "（請選擇）":
        st.info("請先選擇領域")
    else:
        # 檔案選擇
        st.markdown("### 檔案選擇")
        gh_files = list_github_files_by_domain(domain) if src_mode.startswith("GitHub") else []
        file_choices = gh_files
        files = st.multiselect("選擇一個或多個 Excel 檔", file_choices, key="files")

        # 分頁選擇：先讀檔以列出所有分頁供挑選
        all_sheets = []
        for f in files:
            try:
                _dfs = read_one_excel(f)
                for s in _dfs.keys():
                    if s not in all_sheets:
                        all_sheets.append(s)
            except Exception:
                continue
        st.markdown("### 分頁選擇")
        chosen_sheets = st.multiselect("選擇要載入的分頁（不選＝把所選檔案的所有分頁都載入）", all_sheets, key="sheets")
        use_sheet_as_tag = st.checkbox("沒有 Tag 的題目，用分頁名作為 Tag", True, key="use_sheet_as_tag")

        st.write("---")
        st.markdown("### 出題設定")
        mode = st.radio("模式", ["練習", "模擬"], horizontal=True, key="mode")

        # 預先載入一次題庫（僅用來展示可選 Tag）
        tmp_df = assemble_bank(domain, files, chosen_sheets, use_sheet_as_tag) if (domain and files) else pd.DataFrame()
        all_tags = sorted([t for t in tmp_df["Tag"].unique() if str(t).strip() != ""]) if not tmp_df.empty else []
        selected_tags = st.multiselect("選擇章節/標籤（可多選；不選＝全部）", all_tags, key="selected_tags")
        question_count = st.number_input("題數", min_value=1, max_value=max(1, len(tmp_df)), value=min(30, max(1, len(tmp_df))), step=1, key="question_count")
        shuffle = st.checkbox("亂序顯示", True, key="shuffle")

        st.write("---")
        if st.button("開始出題", use_container_width=True) and not tmp_df.empty:
            # 根據條件產生題庫
            df_bank = tmp_df.copy()
            if selected_tags:
                df_bank = df_bank[df_bank["Tag"].isin(selected_tags)].copy()
            if df_bank.empty:
                st.error("沒有符合條件的題目")
            else:
                n = min(question_count, len(df_bank))
                idxs = list(df_bank.index)
                if shuffle:
                    random.shuffle(idxs)
                q_indices = idxs[:n]

                st.session_state.df_bank = df_bank.reset_index(drop=True)
                # 需要把 q_indices 轉為新 df 的 index
                # 因為 reset_index 後索引改變，重取 0..len-1
                q_indices = list(range(n))

                st.session_state.q_indices = q_indices
                st.session_state.current_idx = 0
                st.session_state.answers = {}
                st.session_state.results = {}
                st.session_state.started = True
                st.session_state.start_time = dt.datetime.now()
                st.success(f"已載入題目數：{n}")

# ======================================
# =============== 題目區 ===============
# ======================================

def render_question(qrow: pd.Series, row_idx: int):
    st.markdown(f"### 第 {st.session_state.current_idx+1}/{len(st.session_state.q_indices)} 題")
    st.write(qrow["Question"])

    opts = []
    for k in ["A","B","C","D"]:
        val = str(qrow[f"Option{k}"]).strip()
        if val:
            opts.append((k, f"{k}. {val}"))

    # 取回之前作答
    current_ans = st.session_state.answers.get(row_idx, None)
    choice = st.radio("（單選）", [o[1] for o in opts], key=f"q_{row_idx}", index=None if current_ans is None else [x[0] for x in opts].index(current_ans), label_visibility="collapsed")

    # 轉回 A/B/C/D
    if choice is not None:
        chosen_letter = None
        for k, label in opts:
            if label == choice:
                chosen_letter = k
                break
        if chosen_letter:
            st.session_state.answers[row_idx] = chosen_letter
            if st.session_state.mode == "練習":
                # 立刻判斷
                correct = str(qrow["Answer"]).strip().upper()
                if chosen_letter == correct:
                    st.success(f"✅ 正確（答案：{correct}）")
                    st.session_state.results[row_idx] = True
                else:
                    st.error(f"❌ 錯誤（正確答案：{correct}）")
                    st.session_state.results[row_idx] = False

    # 上一題 / 下一題
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ 上一題", use_container_width=True) and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
    with col2:
        if st.button("下一題 ➡", use_container_width=True) and st.session_state.current_idx < len(st.session_state.q_indices)-1:
            st.session_state.current_idx += 1
            st.rerun()

def evaluate_and_show():
    df = st.session_state.df_bank
    if df.empty:
        st.warning("尚未載入題庫")
        return
    n = len(st.session_state.q_indices)
    rows = []
    for i in range(n):
        row_idx = st.session_state.q_indices[i]
        q = df.iloc[row_idx]
        your = st.session_state.answers.get(row_idx, "")
        correct = str(q["Answer"]).strip().upper()
        ok = (your == correct) and (correct in "ABCD")
        st.session_state.results[row_idx] = bool(ok)
        rows.append({
            "ID": q["ID"],
            "Tag": q["Tag"],
            "Question": q["Question"],
            "A": q["OptionA"], "B": q["OptionB"], "C": q["OptionC"], "D": q["OptionD"],
            "Correct": correct,
            "Your": your or "",
            "Result": "✅" if ok else "❌",
        })
    res_df = pd.DataFrame(rows)
    score = res_df["Result"].eq("✅").sum()
    st.success(f"成績：{score}/{n}（{round(100*score/n,1)} 分）")

    # 只對錯題做 AI 解析
    wrong_mask = res_df["Result"].ne("✅")
    if wrong_mask.any():
        st.write("### AI_Explanation")
        exps = []
        wrong_idx = res_df.index[wrong_mask].tolist()
        for idx in wrong_idx:
            qrow = df.iloc[st.session_state.q_indices[idx]]
            prompt = build_explain_prompt(qrow, res_df.loc[idx,"Your"])
            txt = llm_explain(prompt)
            exps.append(txt)
        res_df.loc[wrong_mask, "AI_Explanation"] = exps
        res_df.loc[~wrong_mask, "AI_Explanation"] = ""
        # 顯示一個精簡表
        st.dataframe(res_df[["ID","Tag","Question","Your","Correct","Result","AI_Explanation"]], use_container_width=True, height=400)
    else:
        st.dataframe(res_df[["ID","Tag","Question","Your","Correct","Result"]], use_container_width=True, height=400)

    # CSV 下載
    csv = res_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載作答結果（CSV）", csv, file_name=f"results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

# ======= 主畫面 =======
if st.session_state.started and not st.session_state.df_bank.empty:
    idx = st.session_state.current_idx
    row_idx = st.session_state.q_indices[idx]
    qrow = st.session_state.df_bank.iloc[row_idx]
    render_question(qrow, row_idx)

    st.write("---")
    if st.session_state.mode == "模擬":
        if st.button("🧾 交卷", type="primary", use_container_width=True):
            evaluate_and_show()
    else:
        # 練習模式：提供「看結果（統整）」按鈕
        if st.button("查看目前作答統整", use_container_width=True):
            evaluate_and_show()
else:
    st.info("請在左側選擇領域、檔案與分頁，然後按下『開始出題』。")
