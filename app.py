# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import re
import json
import base64
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="模擬考與題庫練習", layout="wide", page_icon="📘")
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_BANK_ROOT = REPO_ROOT / "題庫"   # 本機預設題庫資料夾（相對於 repo）
random.seed(42)


# =========================
# 工具：讀 Secrets / Env
# =========================
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        return default


# GitHub 參數（若有就走 GitHub 模式）
GH_TOKEN  = _get_secret("GH_TOKEN",  os.getenv("GH_TOKEN"))
GH_OWNER  = _get_secret("GH_OWNER",  os.getenv("GH_OWNER"))
GH_REPO   = _get_secret("GH_REPO",   os.getenv("GH_REPO"))
GH_BRANCH = _get_secret("GH_BRANCH", os.getenv("GH_BRANCH", "main"))
GH_FOLDER = _get_secret("GH_FOLDER", os.getenv("GH_FOLDER", "題庫"))

# Admin 密碼
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))

# LLM 參數（Gemini）
for k in ["LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"]:
    v = _get_secret(k, os.getenv(k))
    if v:
        os.environ[k] = str(v)


def gh_enabled() -> bool:
    return all([GH_TOKEN, GH_OWNER, GH_REPO, GH_BRANCH, GH_FOLDER])


# 一旦偵測到 GH_*，強制忽略任何本機 BANK_ROOT 類型設定，避免 /Users/... 殘留
if gh_enabled():
    os.environ.pop("BANK_ROOT", None)


# =========================
# Gemini：模型名淨化 + 404 回退
# =========================
def sanitize_gemini_model(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "gemini-1.5-flash"
    n = n.replace("models/", "")
    n = re.sub(r":.*$", "", n)      # 去掉 :latest
    n = re.sub(r"-\d+$", "", n)     # 去掉 -001/-002
    allow = {"gemini-1.5-flash", "gemini-1.5-pro"}
    return n if n in allow else "gemini-1.5-flash"


@st.cache_data(show_spinner=False)
def llm_explain_cached(prompt: str, provider: str, model: str) -> str:
    try:
        if provider.lower() == "gemini":
            import google.generativeai as genai  # 需 requirements: google-generativeai>=0.8.0
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                return "（AI詳解失敗：GEMINI_API_KEY 未設定）"
            genai.configure(api_key=api_key)
            model = sanitize_gemini_model(model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

            def _gen(m: str, p: str):
                g = genai.GenerativeModel(m)
                return g.generate_content(
                    p,
                    generation_config={"temperature": 0.2, "max_output_tokens": 400},
                )

            try:
                resp = _gen(model, prompt)
            except Exception as e:
                # 404 / not found 之類 → 回退到 flash
                if "was not found" in str(e) or "404" in str(e):
                    resp = _gen("gemini-1.5-flash", prompt)
                else:
                    raise
            return (getattr(resp, "text", "") or "").strip()
        else:
            return "（AI詳解失敗：未支援的 LLM_PROVIDER）"
    except Exception as e:
        return f"（AI詳解失敗：{e}）"


# =========================
# GitHub API：讀檔/寫檔
# =========================
def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def gh_list_dir(path: str) -> List[Dict]:
    """列出 GH_FOLDER/path 的內容（目錄/檔案）"""
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    if r.status_code != 200:
        return []
    return r.json()

def gh_read_file(path: str) -> bytes:
    """讀 GH repo 中的檔案（自動處理 large file download_url）"""
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"GitHub 讀檔失敗：{path} ({r.status_code})")
    data = r.json()
    if "content" in data and data.get("encoding") == "base64":
        return base64.b64decode(data["content"])
    # 大檔 → 用 download_url
    download_url = data.get("download_url")
    if not download_url:
        raise RuntimeError(f"GitHub 無 download_url：{path}")
    r2 = requests.get(download_url, timeout=120)
    r2.raise_for_status()
    return r2.content

def gh_get_file_sha(path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def gh_write_file(path: str, content: bytes, message: str) -> bool:
    """建立/更新檔案到 GitHub（需 PAT 有 repo contents:write）"""
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"
    sha = gh_get_file_sha(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": GH_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=gh_headers(), data=json.dumps(payload), timeout=60)
    return 200 <= r.status_code < 300


# =========================
# 資料載入與正規化
# =========================
OPTION_ALIASES = {
    "A": ["A", "OptionA", "選項A", "選項一", "選項1", "答案選項1", "選項甲"],
    "B": ["B", "OptionB", "選項B", "選項二", "選項2", "答案選項2", "選項乙"],
    "C": ["C", "OptionC", "選項C", "選項三", "選項3", "答案選項3", "選項丙"],
    "D": ["D", "OptionD", "選項D", "選項四", "選項4", "答案選項4", "選項丁"],
}
QUESTION_ALIASES = ["Question", "題目", "題幹", "題目內容"]
ANSWER_ALIASES   = ["Answer", "答案", "正解", "正確選項", "正確答案"]
EXPLAIN_ALIASES  = ["Explanation", "詳解", "解析"]
TAG_ALIASES      = ["Tag", "標籤", "章節", "章節/標籤"]
ID_ALIASES       = ["ID", "Id", "題號"]

def find_first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {c.strip(): c for c in df.columns.astype(str)}
    for n in names:
        if n in cols:
            return cols[n]
    # 寬鬆搜尋：忽略空白
    low = {re.sub(r"\s+", "", c.lower()): c for c in df.columns.astype(str)}
    for n in names:
        key = re.sub(r"\s+", "", n.lower())
        if key in low:
            return low[key]
    return None

def normalize_row(row: pd.Series, sheet_tag: Optional[str]) -> Optional[Dict]:
    # 問題
    qcol = find_first_col(row.to_frame().T, QUESTION_ALIASES)
    if not qcol:
        return None
    question = str(row[qcol]).strip()
    if not question:
        return None

    # 選項
    options = {}
    for k, aliases in OPTION_ALIASES.items():
        for a in aliases:
            if a in row.index:
                val = str(row[a]).strip()
                if val and val.lower() not in ["nan", "none"]:
                    options[k] = val
                    break
    # 至少需要兩個選項
    if len(options) < 2:
        return None

    # 答案
    ans_col = find_first_col(row.to_frame().T, ANSWER_ALIASES)
    answer_raw = str(row[ans_col]).strip() if ans_col else ""
    answer = None
    if answer_raw:
        up = answer_raw.upper()
        if up in ["A", "B", "C", "D"]:
            answer = up
        elif up in ["1", "２", "１", "①", "1️⃣"]:
            answer = "A"
        elif up in ["2", "②", "2️⃣"]:
            answer = "B"
        elif up in ["3", "③", "3️⃣"]:
            answer = "C"
        elif up in ["4", "④", "4️⃣"]:
            answer = "D"
        else:
            # 若答案填的是選項文字 → 轉成對應 A/B/C/D
            for k, v in options.items():
                if v == answer_raw:
                    answer = k
                    break

    # Tag
    tag_col = find_first_col(row.to_frame().T, TAG_ALIASES)
    tag = str(row[tag_col]).strip() if (tag_col and str(row[tag_col]).strip()) else (sheet_tag or "")

    # ID
    id_col = find_first_col(row.to_frame().T, ID_ALIASES)
    rid = str(row[id_col]).strip() if id_col else ""

    # 詳解（若題庫本身有）
    exp_col = find_first_col(row.to_frame().T, EXPLAIN_ALIASES)
    expl = str(row[exp_col]).strip() if exp_col else ""

    return {
        "id": rid,
        "question": question,
        "options": options,   # dict: {"A": "...", "B": "..."}
        "answer": answer,     # 可能為 None → 視為題庫未標答案
        "tag": tag,
        "explain": expl,
    }

def read_excel_bytes(xls_bytes: bytes, filename: str, selected_sheets: Optional[List[str]]) -> List[Dict]:
    buf = io.BytesIO(xls_bytes)
    # 自動選 engine（.xls -> xlrd）
    engine = "xlrd" if filename.lower().endswith(".xls") else None
    xf = pd.ExcelFile(buf, engine=engine)
    sheets = selected_sheets or xf.sheet_names
    results: List[Dict] = []
    for s in sheets:
        try:
            df = xf.parse(s)
        except Exception:
            continue
        sheet_tag = s  # 可當預設 Tag
        # 將每列轉為標準題目結構
        for _, r in df.iterrows():
            item = normalize_row(r, sheet_tag=sheet_tag)
            if item:
                # 填入來源資訊（方便回溯）
                item["source_file"] = filename
                item["source_sheet"] = s
                # 若缺 ID，用「檔名_分頁_流水號」
                if not item["id"]:
                    item["id"] = f"{Path(filename).stem}:{s}:{_}"
                results.append(item)
    return results


@st.cache_data(show_spinner=True)
def load_bank_from_github(domain: str, files: List[str], sheet_map: Dict[str, List[str]]) -> List[Dict]:
    """files 是該 domain 底下的檔名清單（不含路徑）"""
    all_items: List[Dict] = []
    for fname in files:
        rel = f"{GH_FOLDER}/{domain}/{fname}"
        try:
            xbytes = gh_read_file(rel)
            items = read_excel_bytes(xbytes, fname, sheet_map.get(fname))
            all_items.extend(items)
        except Exception as e:
            st.warning(f"讀取 {rel} 失敗：{e}")
    return all_items

@st.cache_data(show_spinner=True)
def load_bank_from_local(domain: str, files: List[str], sheet_map: Dict[str, List[str]]) -> List[Dict]:
    all_items: List[Dict] = []
    base = LOCAL_BANK_ROOT / domain
    for fname in files:
        p = base / fname
        if not p.exists():
            st.warning(f"找不到檔案：{p}")
            continue
        try:
            xbytes = p.read_bytes()
            items = read_excel_bytes(xbytes, fname, sheet_map.get(fname))
            all_items.extend(items)
        except Exception as e:
            st.warning(f"讀取 {p} 失敗：{e}")
    return all_items


# =========================
# UI：側邊欄
# =========================
def sidebar_source_and_admin():
    st.sidebar.header("資料來源與管理")
    if gh_enabled():
        src_text = f"來源模式：**GitHub / {GH_FOLDER}**"
    else:
        src_text = f"來源模式：**本機 / {LOCAL_BANK_ROOT.name}**"
    st.sidebar.caption(src_text)

    # 管理模式切換
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if st.sidebar.button("啟用管理模式" if not st.session_state.is_admin else "關閉管理模式", use_container_width=True):
        if not st.session_state.is_admin:
            if ADMIN_PASSWORD:
                pw = st.sidebar.text_input("輸入管理密碼", type="password", key="__admin_pw__", value="", placeholder="Admin password", label_visibility="collapsed")
                st.stop()
            else:
                st.session_state.is_admin = True
                st.rerun()
        else:
            st.session_state.is_admin = False
            st.rerun()

    if st.session_state.is_admin:
        st.sidebar.success("管理模式已啟用")
        if gh_enabled():
            st.sidebar.markdown("**上傳 Excel 到 GitHub**")
            domain = st.session_state.get("domain") or ""
            up = st.sidebar.file_uploader("選擇 Excel 檔（上傳後會 commit 到目前領域資料夾）", type=["xlsx", "xls"], accept_multiple_files=False)
            if up and domain:
                content = up.read()
                path = f"{GH_FOLDER}/{domain}/{up.name}"
                ok = gh_write_file(path, content, message=f"upload {up.name} to {domain}")
                if ok:
                    st.sidebar.success(f"已更新：{path}")
                    st.cache_data.clear()
                else:
                    st.sidebar.error("上傳失敗，請檢查 GH_TOKEN 權限（需 contents:write）")
        else:
            st.sidebar.info("本機模式不提供上傳 GitHub。")


def sidebar_pick_domain_files_sheets() -> Tuple[str, List[str], Dict[str, List[str]], bool]:
    st.sidebar.header("領域選擇")

    if gh_enabled():
        # 列出 GH_FOLDER 下的子資料夾
        roots = gh_list_dir(GH_FOLDER or "題庫")
        domains = [d["name"] for d in roots if d.get("type") == "dir"]
    else:
        if not LOCAL_BANK_ROOT.exists():
            st.sidebar.error(f"找不到根目錄：{LOCAL_BANK_ROOT}")
            return "", [], {}, False
        domains = sorted([p.name for p in LOCAL_BANK_ROOT.iterdir() if p.is_dir()])

    domain = st.sidebar.selectbox("選擇領域", options=domains or ["（無資料夾）"], index=0)
    st.session_state.domain = domain

    # 檔案選擇
    st.sidebar.header("檔案選擇")
    if gh_enabled():
        files_json = gh_list_dir(f"{GH_FOLDER}/{domain}")
        excel_files = [f["name"] for f in files_json if f.get("type") == "file" and f["name"].lower().endswith((".xlsx", ".xls"))]
    else:
        p = LOCAL_BANK_ROOT / domain
        excel_files = sorted([x.name for x in p.glob("*.xls*")])

    picked_files = st.sidebar.multiselect("選擇一個或多個 Excel 檔", options=excel_files, default=excel_files[:1])

    # 分頁選擇（逐檔）
    st.sidebar.header("分頁選擇")
    sheet_map: Dict[str, List[str]] = {}
    for fname in picked_files:
        try:
            if gh_enabled():
                b = gh_read_file(f"{GH_FOLDER}/{domain}/{fname}")
            else:
                b = (LOCAL_BANK_ROOT / domain / fname).read_bytes()
            xf = pd.ExcelFile(io.BytesIO(b), engine=("xlrd" if fname.endswith(".xls") else None))
            sheets = xf.sheet_names
            sel = st.sidebar.multiselect(f"{Path(fname).stem}", options=sheets, default=sheets, key=f"__sheets__{fname}")
            sheet_map[fname] = sel
        except Exception as e:
            st.sidebar.warning(f"{fname} 讀取分頁失敗：{e}")

    use_sheet_as_tag = st.sidebar.checkbox("沒有 Tag 的題目，用分頁名作為 Tag", value=True)

    return domain, picked_files, sheet_map, use_sheet_as_tag


# =========================
# 題庫載入 + 過濾
# =========================
def assemble_bank(domain: str, files: List[str], sheet_map: Dict[str, List[str]], use_sheet_tag: bool) -> pd.DataFrame:
    if not domain or not files:
        return pd.DataFrame()

    if gh_enabled():
        items = load_bank_from_github(domain, files, sheet_map)
    else:
        items = load_bank_from_local(domain, files, sheet_map)

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    if use_sheet_tag:
        df["tag"] = df["tag"].fillna("").replace("", df["source_sheet"])
    else:
        df["tag"] = df["tag"].fillna("")

    # 只留有兩個以上選項的題目
    df = df[df["options"].apply(lambda d: isinstance(d, dict) and len(d) >= 2)]
    df = df.reset_index(drop=True)
    return df


# =========================
# 出題 UI 與流程
# =========================
def show_question(qidx: int, df: pd.DataFrame, mode: str, state_key_prefix: str = "q") -> None:
    row = df.iloc[qidx]
    qid = row["id"]
    options: Dict[str, str] = row["options"]
    answer = row.get("answer")
    question = row["question"]
    tag = row.get("tag", "")

    st.subheader(f"第 {qidx+1}/{len(df)} 題")
    st.markdown(f"**{question}**")
    if tag:
        st.caption(f"Tag：{tag}")

    # 選項以固定順序 A-D 顯示（只顯示存在的）
    order = [k for k in ["A","B","C","D"] if k in options]
    labels = [f"{k}. {options[k]}" for k in order]

    # 使用 qid 保持作答狀態
    key = f"{state_key_prefix}_{qid}"
    picked = st.radio("選擇答案", options=order, format_func=lambda k: f"{k}. {options[k]}", index=None, key=key)

    # 練習模式：即時給判斷與詳解
    if mode == "練習" and picked:
        if answer and picked == answer:
            st.success(f"✅ 正確！答案：{picked}")
        elif answer:
            st.error(f"❌ 錯誤，正解：{answer}")
        else:
            st.info("此題題庫未標示正解，僅記錄作答。")

        # 顯示題庫已有詳解，否則用 LLM
        builtin_exp = str(row.get("explain") or "").strip()
        if builtin_exp:
            with st.expander("題庫詳解", expanded=True):
                st.write(builtin_exp)
        else:
            provider = os.getenv("LLM_PROVIDER", "gemini")
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            prompt = make_explain_prompt(question, options, answer, picked)
            ai_text = llm_explain_cached(prompt, provider, model)
            with st.expander("AI 詳解", expanded=True):
                st.write(ai_text)


def make_explain_prompt(question: str, options: Dict[str, str], answer: Optional[str], picked: Optional[str]) -> str:
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    ans_text = answer if answer else "（題庫未標答案，請依專業判斷後給出最可能的正解）"
    picked_text = picked or "（尚未作答）"
    prompt = f"""
你是一位保險/金融考題的專業出題與解析老師。請用繁體中文，針對這題出「精簡但清楚」的解析，條列要點與關鍵概念、避免贅詞。

題目：
{question}

選項：
{opt_text}

正確答案（若未提供請先推論出最合適的選項再解釋理由）：
{ans_text}

考生選擇：
{picked_text}

請輸出：
1) 正確選項（若題庫未標，請先判斷）
2) 解析重點（條列）
3) 爭點/易錯點提示（條列）
"""
    return textwrap.dedent(prompt).strip()


# =========================
# 主程式
# =========================
def main():
    st.markdown("## 📘 模擬考與題庫練習")

    # 來源 / 管理
    sidebar_source_and_admin()

    # 選擇領域/檔案/分頁
    domain, files, sheet_map, use_sheet_tag = sidebar_pick_domain_files_sheets()

    # 出題設定
    st.sidebar.header("出題設定")
    mode = st.sidebar.radio("模式", options=["練習", "模擬"], index=0, horizontal=True)
    # 載入題庫
    df_bank = assemble_bank(domain, files, sheet_map, use_sheet_tag)

    if df_bank.empty:
        st.info("請先在左側選擇領域 → 檔案 → 分頁。")
        return

    st.success(f"已載入題目數：{len(df_bank)}")

    # Tag 過濾
    all_tags = sorted([t for t in df_bank["tag"].astype(str).unique() if str(t).strip() != ""])
    picked_tags = st.sidebar.multiselect("選擇章節/標籤（可多選；不選＝全部）", options=all_tags, default=[])
    if picked_tags:
        df_use = df_bank[df_bank["tag"].isin(picked_tags)].reset_index(drop=True)
    else:
        df_use = df_bank

    # 題數與亂序
    default_n = min(30, len(df_use))
    n_questions = st.sidebar.number_input("題數", min_value=1, max_value=len(df_use), value=default_n, step=1)
    shuffle = st.sidebar.checkbox("亂序顯示", value=True)

    # 抽題
    if shuffle:
        df_use = df_use.sample(frac=1.0, random_state=None).reset_index(drop=True)
    df_use = df_use.iloc[:n_questions].reset_index(drop=True)

    # 顯示 LLM 狀態（除錯）
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model_shown = sanitize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    st.caption(f"偵測供應者：**{provider}** / 模型：**{model_shown}**")

    # 狀態初始化
    if "cur_idx" not in st.session_state:
        st.session_state.cur_idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}  # qid -> {"picked": "A"}
    if "paper_ids" not in st.session_state:
        st.session_state.paper_ids = list(df_use["id"])  # 用於模擬考的固定題序

    # 重新對齊題序（切換設定後）
    st.session_state.paper_ids = list(df_use["id"])

    # 主要畫面
    if mode == "練習":
        i = st.session_state.cur_idx
        show_question(i, df_use, mode="練習", state_key_prefix="prac")

        cols = st.columns(3)
        with cols[0]:
            if st.button("⬅️ 上一題", use_container_width=True, disabled=(i == 0)):
                st.session_state.cur_idx = max(0, i - 1)
                st.rerun()
        with cols[1]:
            if st.button("🔄 重新整理本題", use_container_width=True):
                st.rerun()
        with cols[2]:
            if st.button("➡️ 下一題", use_container_width=True, disabled=(i >= len(df_use)-1)):
                st.session_state.cur_idx = min(len(df_use)-1, i + 1)
                st.rerun()

    else:  # 模擬考
        # 作答紀錄
        i = st.session_state.cur_idx
        row = df_use.iloc[i]
        qid = row["id"]
        question = row["question"]
        options: Dict[str, str] = row["options"]

        st.subheader(f"第 {i+1}/{len(df_use)} 題（模擬考）")
        st.write(question)
        order = [k for k in ["A","B","C","D"] if k in options]

        key = f"exam_{qid}"
        picked = st.radio("選擇答案", options=order, format_func=lambda k: f"{k}. {options[k]}", index=None, key=key)
        if picked:
            st.session_state.answers[qid] = {"picked": picked}

        cols = st.columns(3)
        with cols[0]:
            if st.button("⬅️ 上一題", use_container_width=True, disabled=(i == 0)):
                st.session_state.cur_idx = max(0, i - 1)
                st.rerun()
        with cols[1]:
            if st.button("➡️ 下一題", use_container_width=True, disabled=(i >= len(df_use)-1)):
                st.session_state.cur_idx = min(len(df_use)-1, i + 1)
                st.rerun()
        with cols[2]:
            if st.button("🧾 交卷", type="primary", use_container_width=True):
                # 計分
                score = 0
                rows = []
                for _, r in df_use.iterrows():
                    qid = r["id"]
                    ans = r.get("answer")
                    opt = r["options"]
                    picked = st.session_state.answers.get(qid, {}).get("picked")
                    result = (picked == ans) if ans else None
                    if result:
                        score += 1

                    # AI 詳解（僅錯題或未標正解）
                    builtin = str(r.get("explain") or "").strip()
                    ai_text = ""
                    if not builtin:
                        prompt = make_explain_prompt(r["question"], opt, ans, picked)
                        ai_text = llm_explain_cached(prompt, provider, model_shown)

                    rows.append({
                        "ID": qid,
                        "Tag": r.get("tag", ""),
                        "Question": r["question"],
                        "OptionA": opt.get("A",""),
                        "OptionB": opt.get("B",""),
                        "OptionC": opt.get("C",""),
                        "OptionD": opt.get("D",""),
                        "Answer": ans or "",
                        "YourAnswer": picked or "",
                        "Result": "" if result is None else ("O" if result else "X"),
                        "Builtin_Explanation": builtin,
                        "AI_Explanation": ai_text if not builtin else "",
                        "SourceFile": r.get("source_file",""),
                        "SourceSheet": r.get("source_sheet",""),
                    })

                total = len(df_use)
                st.success(f"成績：{score}/{total}（{round(score*100/total,1)} 分）")

                out_df = pd.DataFrame(rows)
                st.dataframe(out_df, use_container_width=True, height=400)

                # 下載 CSV
                csv = out_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下載作答結果（CSV）", data=csv, file_name="exam_result.csv", mime="text/csv")

                # 顯示 AI 詳解（僅錯題/未標正解才有）
                wrong_df = out_df[(out_df["Result"] == "X") | (out_df["Result"] == "")]
                if not wrong_df.empty:
                    st.markdown("---")
                    st.markdown("### 錯題與 AI 詳解")
                    for _, rr in wrong_df.iterrows():
                        with st.expander(f"題目：{rr['Question'][:50]}..."):
                            st.write(f"正解：{rr['Answer'] or '（未標）'} | 你的答案：{rr['YourAnswer'] or '（未作答）'}")
                            if rr["Builtin_Explanation"]:
                                st.markdown("**題庫詳解：**")
                                st.write(rr["Builtin_Explanation"])
                            if rr["AI_Explanation"]:
                                st.markdown("**AI 詳解：**")
                                st.write(rr["AI_Explanation"])

    st.markdown("---")
    st.caption("若看到 /Users/... 找不到，請確認已切換到 GitHub 模式並移除 BANK_ROOT；或把題庫放在 repo 的『題庫/』內。")


if __name__ == "__main__":
    main()
