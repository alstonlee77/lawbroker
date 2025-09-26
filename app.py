# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import re
import json
import base64
import textwrap
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="模擬考與題庫練習", layout="wide", page_icon="📘")
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_BANK_ROOT = REPO_ROOT / "題庫"   # 本機預設題庫資料夾（相對於 repo）
random.seed(42)

# ---- 兼容 rerun API ----
def _rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()  # type: ignore[attr-defined]
    else:
        st.rerun()

# =========================
# Secrets / Env
# =========================
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        return default

# GitHub（若有就啟用雲端題庫）
GH_TOKEN  = _get_secret("GH_TOKEN",  os.getenv("GH_TOKEN"))
GH_OWNER  = _get_secret("GH_OWNER",  os.getenv("GH_OWNER"))
GH_REPO   = _get_secret("GH_REPO",   os.getenv("GH_REPO"))
GH_BRANCH = _get_secret("GH_BRANCH", os.getenv("GH_BRANCH", "main"))
GH_FOLDER = _get_secret("GH_FOLDER", os.getenv("GH_FOLDER", "題庫"))

# Admin 密碼
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))

# LLM（Gemini）
for k in ["LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"]:
    v = _get_secret(k, os.getenv(k))
    if v:
        os.environ[k] = str(v)

def gh_enabled() -> bool:
    return all([GH_TOKEN, GH_OWNER, GH_REPO, GH_BRANCH, GH_FOLDER])

# 若偵測到 GH_*，避免誤回本機模式
if gh_enabled():
    os.environ.pop("BANK_ROOT", None)

# =========================
# Gemini：模型名淨化 + 404 回退
# =========================
def sanitize_gemini_model(name: str) -> str:
    """
    將各種形式的名稱（models/xxx, :latest, -001/-002）淨化成正式名。
    預設使用 gemini-2.5-flash；若不在白名單就回到 2.5-flash。
    """
    n = (name or "").strip()
    if not n:
        return "gemini-2.5-flash"

    n = n.replace("models/", "")
    n = re.sub(r":.*$", "", n)      # :latest
    n = re.sub(r"-\d+$", "", n)     # -001 / -002

    allow = {
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    }
    return n if n in allow else "gemini-2.5-flash"

@st.cache_data(show_spinner=False)
def llm_explain_cached(prompt: str, provider: str, model: str) -> str:
    """
    以 Gemini 產生詳解；若 2.5 無權限/404，自動退回 1.5-flash。
    """
    try:
        if provider.lower() == "gemini":
            import google.generativeai as genai  # google-generativeai>=0.8.0
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                return "（AI詳解失敗：GEMINI_API_KEY 未設定）"
            genai.configure(api_key=api_key)
            model = sanitize_gemini_model(model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

            def _gen(m: str, p: str):
                g = genai.GenerativeModel(m)
                return g.generate_content(
                    p,
                    generation_config={"temperature": 0.2, "max_output_tokens": 400},
                )

            try:
                resp = _gen(model, prompt)
            except Exception as e:
                # 2.5 沒權限或找不到 → 退回 1.5
                if "was not found" in str(e) or "404" in str(e):
                    try:
                        resp = _gen("gemini-1.5-flash", prompt)
                    except Exception as e2:
                        raise e2
                else:
                    raise
            return (getattr(resp, "text", "") or "").strip()
        else:
            return "（AI詳解失敗：未支援的 LLM_PROVIDER）"
    except Exception as e:
        return f"（AI詳解失敗：{e}）"

# =========================
# GitHub API
# =========================
def gh_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def gh_list_dir(path: str) -> List[Dict]:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=30)
    if r.status_code != 200:
        return []
    return r.json()

def gh_read_file(path: str) -> bytes:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=gh_headers(), timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"GitHub 讀檔失敗：{path} ({r.status_code})")
    data = r.json()
    if "content" in data and data.get("encoding") == "base64":
        return base64.b64decode(data["content"])
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
# 欄位對應 / 題庫讀取
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
    low = {re.sub(r"\s+", "", c.lower()): c for c in df.columns.astype(str)}
    for n in names:
        key = re.sub(r"\s+", "", n.lower())
        if key in low:
            return low[key]
    return None

def normalize_row(row: pd.Series, sheet_tag: Optional[str]) -> Optional[Dict]:
    qcol = find_first_col(row.to_frame().T, QUESTION_ALIASES)
    if not qcol:
        return None
    question = str(row[qcol]).strip()
    if not question:
        return None

    options = {}
    for k, aliases in OPTION_ALIASES.items():
        for a in aliases:
            if a in row.index:
                val = str(row[a]).strip()
                if val and val.lower() not in ["nan", "none"]:
                    options[k] = val
                    break
    if len(options) < 2:
        return None

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
            for k, v in options.items():
                if v == answer_raw:
                    answer = k
                    break

    tag_col = find_first_col(row.to_frame().T, TAG_ALIASES)
    tag = str(row[tag_col]).strip() if (tag_col and str(row[tag_col]).strip()) else (sheet_tag or "")

    id_col = find_first_col(row.to_frame().T, ID_ALIASES)
    rid = str(row[id_col]).strip() if id_col else ""

    exp_col = find_first_col(row.to_frame().T, EXPLAIN_ALIASES)
    expl = str(row[exp_col]).strip() if exp_col else ""

    return {
        "id": rid,
        "question": question,
        "options": options,
        "answer": answer,
        "tag": tag,
        "explain": expl,
    }

def read_excel_bytes(xls_bytes: bytes, filename: str, selected_sheets: Optional[List[str]]) -> List[Dict]:
    buf = io.BytesIO(xls_bytes)
    engine = "xlrd" if filename.lower().endswith(".xls") else None
    xf = pd.ExcelFile(buf, engine=engine)
    sheets = selected_sheets or xf.sheet_names
    results: List[Dict] = []
    for s in sheets:
        try:
            df = xf.parse(s)
        except Exception:
            continue
        sheet_tag = s
        for idx, r in df.iterrows():
            item = normalize_row(r, sheet_tag=sheet_tag)
            if item:
                item["source_file"] = filename
                item["source_sheet"] = s
                if not item["id"]:
                    item["id"] = f"{Path(filename).stem}:{s}:{idx}"
                results.append(item)
    return results

@st.cache_data(show_spinner=True)
def load_bank_from_github(domain: str, files: List[str], sheet_map: Dict[str, List[str]]) -> List[Dict]:
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
        df["tag"] = df.get("tag", "").astype(str)
        df["source_sheet"] = df.get("source_sheet", "").astype(str)
        df["tag"] = df["tag"].replace({"nan": ""})
        mask = df["tag"].isna() | (df["tag"].str.strip() == "")
        df.loc[mask, "tag"] = df.loc[mask, "source_sheet"]
    else:
        df["tag"] = df.get("tag", "").fillna("").astype(str)

    df = df[df["options"].apply(lambda d: isinstance(d, dict) and len(d) >= 2)]
    df = df.reset_index(drop=True)
    return df

# =========================
# 側邊：管理與來源
# =========================
def sidebar_source_and_admin():
    st.sidebar.header("資料來源與管理")
    if gh_enabled():
        st.sidebar.caption(f"來源模式：**GitHub / {GH_FOLDER}**")
    else:
        st.sidebar.caption(f"來源模式：**本機 / {LOCAL_BANK_ROOT.name}**")

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        if st.sidebar.button("啟用管理模式", use_container_width=True):
            if ADMIN_PASSWORD:
                st.session_state.__await_pw__ = True
            else:
                st.session_state.is_admin = True
            _rerun()
    else:
        if st.sidebar.button("關閉管理模式", use_container_width=True):
            st.session_state.is_admin = False
            _rerun()

    if st.session_state.get("__await_pw__", False):
        pw = st.sidebar.text_input("輸入管理密碼", type="password")
        if st.sidebar.button("登入"):
            if pw == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.session_state.__await_pw__ = False
                _rerun()
            else:
                st.sidebar.error("密碼錯誤。")

    if st.session_state.is_admin and gh_enabled():
        st.sidebar.success("管理模式已啟用（GitHub）")
        domain = st.session_state.get("domain") or ""
        up = st.sidebar.file_uploader("上傳 Excel 到目前領域", type=["xlsx", "xls"])
        if up and domain:
            content = up.read()
            path = f"{GH_FOLDER}/{domain}/{up.name}"
            ok = gh_write_file(path, content, f"upload {up.name} to {domain}")
            if ok:
                st.sidebar.success(f"已更新：{path}")
                st.cache_data.clear()
            else:
                st.sidebar.error("上傳失敗，請檢查 Token 權限（contents:write）。")
    elif st.session_state.is_admin:
        st.sidebar.info("本機模式不提供上傳 GitHub。")

# =========================
# 側邊：選擇領域/檔案/分頁
# =========================
def sidebar_pick_domain_files_sheets() -> Tuple[str, List[str], Dict[str, List[str]], bool]:
    st.sidebar.header("領域選擇")
    if gh_enabled():
        roots = gh_list_dir(GH_FOLDER or "題庫")
        domains = [d["name"] for d in roots if d.get("type") == "dir"]
    else:
        if not LOCAL_BANK_ROOT.exists():
            st.sidebar.error(f"找不到根目錄：{LOCAL_BANK_ROOT}")
            return "", [], {}, False
        domains = sorted([p.name for p in LOCAL_BANK_ROOT.iterdir() if p.is_dir()])

    domain = st.sidebar.selectbox("選擇領域", options=domains or ["（無）"])
    st.session_state.domain = domain

    st.sidebar.header("檔案選擇")
    if gh_enabled():
        files_json = gh_list_dir(f"{GH_FOLDER}/{domain}")
        excel_files = [f["name"] for f in files_json if f.get("type") == "file" and f["name"].lower().endswith((".xlsx", ".xls"))]
    else:
        p = LOCAL_BANK_ROOT / domain
        excel_files = sorted([x.name for x in p.glob("*.xls*")])

    picked_files = st.sidebar.multiselect("選擇一個或多個 Excel 檔", options=excel_files, default=excel_files[:1])

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
            sel = st.sidebar.multiselect(Path(fname).stem, options=sheets, default=sheets, key=f"__sheets__{fname}")
            sheet_map[fname] = sel
        except Exception as e:
            st.sidebar.warning(f"{fname} 讀取分頁失敗：{e}")

    use_sheet_as_tag = st.sidebar.checkbox("沒有 Tag 的題目，用分頁名作為 Tag", value=True)
    return domain, picked_files, sheet_map, use_sheet_as_tag

# =========================
# 顯示題目 & Prompt
# =========================
def make_explain_prompt(question: str, options: Dict[str, str], answer: Optional[str], picked: Optional[str]) -> str:
    opt_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    ans_text = answer if answer else "（題庫未標答案，請依專業判斷後給出最可能正解）"
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

def show_question(qidx: int, df: pd.DataFrame, mode: str, state_key_prefix: str = "q") -> None:
    row = df.iloc[qidx]
    qid = row["id"]
    options: Dict[str, str] = row["options"]
    answer = row.get("answer")
    question = row["question"]
    tag = row.get("tag", "")

    st.subheader(f"第 {qidx+1}/{len(df)} 題")
    if tag:
        st.caption(f"Tag：{tag}")
    st.markdown(f"**{question}**")

    order = [k for k in ["A","B","C","D"] if k in options]
    key = f"{state_key_prefix}_{qid}"
    picked = st.radio("選擇答案", options=order, format_func=lambda k: f"{k}. {options[k]}", index=None, key=key)

    # 練習模式即時回饋
    if mode == "練習" and picked:
        if answer and picked == answer:
            st.success(f"✅ 正確！答案：{picked}")
        elif answer:
            st.error(f"❌ 錯誤，正解：{answer}")
        else:
            st.info("此題題庫未標示正解，僅記錄作答。")

        builtin_exp = str(row.get("explain") or "").strip()
        if builtin_exp:
            with st.expander("題庫詳解", expanded=True):
                st.write(builtin_exp)
        else:
            provider = os.getenv("LLM_PROVIDER", "gemini")
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            prompt = make_explain_prompt(question, options, answer, picked)
            ai_text = llm_explain_cached(prompt, provider, model)
            with st.expander("AI 詳解", expanded=True):
                st.write(ai_text)

# =========================
# 主程式
# =========================
def main():
    st.markdown("## 📘 模擬考與題庫練習")

    # 側邊：來源/管理
    sidebar_source_and_admin()

    # 側邊：選擇資料
    domain, files, sheet_map, use_sheet_tag = sidebar_pick_domain_files_sheets()

    # 側邊：出題設定
    st.sidebar.header("出題設定")
    mode = st.sidebar.radio("模式", options=["練習", "模擬"], index=0)

    plan_key = json.dumps({
        "domain": domain,
        "files": files,
        "sheets": sheet_map,
        "use_sheet_tag": use_sheet_tag,
        "mode": mode,
    }, ensure_ascii=False, sort_keys=True)

    n_default = 30
    n_pick = st.sidebar.number_input("題數", min_value=1, max_value=500, value=5 if mode=="模擬" else n_default, step=1)
    do_shuffle = st.sidebar.checkbox("亂序顯示", value=True)

    # 若任何條件改變 → 重置開始狀態
    if st.session_state.get("__plan_key__") != plan_key:
        st.session_state.__plan_key__ = plan_key
        st.session_state.started = False
        st.session_state.submitted = False
        st.session_state.cur_idx = 0
        st.session_state.answers = {}
        st.session_state.paper_ids = []
        st.session_state.result_df = None

    # 顯示供應者/模型（資訊）
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model_shown = sanitize_gemini_model(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    st.caption(f"偵測供應者：**{provider}** / 模型：**{model_shown}**")

    # ======= 尚未開始：顯示「開始出題」 =======
    if not st.session_state.get("started", False):
        st.info("請先在左側完成選擇，然後按下「開始出題」。")
        if st.button("🚀 開始出題", type="primary", use_container_width=True):
            df_bank = assemble_bank(domain, files, sheet_map, use_sheet_tag)
            if df_bank.empty:
                st.error("載入題庫失敗或為空，請確認選擇。")
                return

            all_tags = sorted([t for t in df_bank["tag"].astype(str).unique() if str(t).strip() != ""])
            st.session_state.__all_tags__ = all_tags

            picked_tags = st.sidebar.multiselect("選擇章節/標籤（可多選；不選＝全部）", options=all_tags, default=[])

            if picked_tags:
                df_use = df_bank[df_bank["tag"].isin(picked_tags)].reset_index(drop=True)
            else:
                df_use = df_bank

            if do_shuffle:
                df_use = df_use.sample(frac=1.0, random_state=None).reset_index(drop=True)
            df_use = df_use.iloc[:min(n_pick, len(df_use))].reset_index(drop=True)

            if df_use.empty:
                st.error("過濾後沒有題目，請調整條件。")
                return

            st.session_state.paper_df = df_use
            st.session_state.paper_ids = list(df_use["id"])
            st.session_state.cur_idx = 0
            st.session_state.answers = {}
            st.session_state.submitted = False
            st.session_state.started = True
            _rerun()
        return

    # ======= 已開始：題目流程 =======
    df_use: pd.DataFrame = st.session_state.get("paper_df", pd.DataFrame())
    if df_use.empty:
        st.warning("目前沒有題目，請重新按「開始出題」。")
        st.session_state.started = False
        return

    st.success(f"本次抽題數：{len(df_use)}")
    if "__all_tags__" in st.session_state and st.session_state.__all_tags__:
        st.caption("可用標籤（資訊）： " + "、".join(st.session_state.__all_tags__))

    i = st.session_state.get("cur_idx", 0)
    i = max(0, min(i, len(df_use)-1))
    st.session_state.cur_idx = i

    if mode == "練習":
        show_question(i, df_use, mode="練習", state_key_prefix="prac")

        nav = st.columns([1,1,1,1])
        with nav[0]:
            if st.button("⬅️ 上一題", use_container_width=True, disabled=(i == 0)):
                st.session_state.cur_idx = max(0, i-1); _rerun()
        with nav[1]:
            if st.button("🔄 重新抽題", use_container_width=True):
                st.session_state.started = False; _rerun()
        with nav[2]:
            if st.button("➡️ 下一題", use_container_width=True, disabled=(i >= len(df_use)-1)):
                st.session_state.cur_idx = min(len(df_use)-1, i+1); _rerun()
        with nav[3]:
            pass

    else:  # 模擬考
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

        nav = st.columns([1,1,1,1])
        with nav[0]:
            if st.button("⬅️ 上一題", use_container_width=True, disabled=(i == 0)):
                st.session_state.cur_idx = max(0, i-1); _rerun()
        with nav[1]:
            if st.button("➡️ 下一題", use_container_width=True, disabled=(i >= len(df_use)-1)):
                st.session_state.cur_idx = min(len(df_use)-1, i+1); _rerun()
        with nav[2]:
            if st.button("🧾 交卷", type="primary", use_container_width=True, disabled=st.session_state.get("submitted", False)):
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
                st.session_state.submitted = True
                st.session_state.score = score
                st.session_state.total = total
                st.session_state.result_df = pd.DataFrame(rows)
                _rerun()
        with nav[3]:
            if st.button("🔁 重新開始", use_container_width=True):
                st.session_state.started = False
                st.session_state.submitted = False
                _rerun()

        # 只有交卷後才顯示成績與表格
        if st.session_state.get("submitted", False):
            score = st.session_state.get("score", 0)
            total = st.session_state.get("total", len(df_use))
            st.success(f"成績：{score}/{total}（{round(score*100/total,1)} 分）")

            out_df = st.session_state.get("result_df")
            if out_df is not None:
                st.dataframe(out_df, use_container_width=True, height=400)
                csv = out_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下載作答結果（CSV）", data=csv, file_name="exam_result.csv", mime="text/csv")

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
    st.caption("若顯示 /Users/... 找不到表示在本機模式；要用 GitHub 題庫請設定 GH_* 並確保題庫在 repo 的『題庫/』資料夾中。")

if __name__ == "__main__":
    main()
