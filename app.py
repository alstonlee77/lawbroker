# app.py — Gemini 版（Google）：
# - 練習模式：即時判斷＋僅錯題 AI 詳解（Gemini）
# - 模考：交卷出分數＋只錯題 AI 詳解＋CSV 下載
# - 題庫：先選領域（人身/外幣/投資型）→ 檔案（多選）→ 分頁（多選）
# - 來源：GitHub 或 本機 /題庫
# - 欄位相容：OptionA~E、答案選項1~5、A~E/1~5/甲乙丙丁戊/全形、星號(*)標正解

from __future__ import annotations
import os, io, re, json, base64, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

# ============== 基本設定 ==============
st.set_page_config(page_title="模擬考與題庫練習", layout="wide", page_icon="📘")
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_BANK_ROOT = Path(os.getenv("BANK_ROOT", REPO_ROOT / "題庫"))

def _get_secret(k: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(k, default)  # type: ignore[attr-defined]
    except Exception:
        return default

# GitHub 參數（有設就會開啟 GitHub 模式）
GH_TOKEN   = _get_secret("GH_TOKEN", os.getenv("GH_TOKEN"))
GH_OWNER   = _get_secret("GH_OWNER", os.getenv("GH_OWNER"))
GH_REPO    = _get_secret("GH_REPO",  os.getenv("GH_REPO"))
GH_BRANCH  = _get_secret("GH_BRANCH", os.getenv("GH_BRANCH", "main"))
GH_FOLDER  = _get_secret("GH_FOLDER", os.getenv("GH_FOLDER", "題庫"))

# Admin 密碼
ADMIN_PASSWORD = _get_secret("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", ""))

# LLM 參數（把 secrets 注入 os.environ，便於統一取用）
for key in ["LLM_PROVIDER","GEMINI_API_KEY","GEMINI_MODEL",
            "OPENAI_API_KEY","OPENAI_MODEL","OLLAMA_MODEL","OLLAMA_ENDPOINT"]:
    val = _get_secret(key)
    if val and not os.getenv(key):
        os.environ[key] = str(val)

# ============== 小工具 ==============
def gh_enabled() -> bool:
    return bool(GH_TOKEN and GH_OWNER and GH_REPO and GH_BRANCH)

def info(msg: str): st.info(msg, icon="ℹ️")
def warn(msg: str): st.warning(msg, icon="⚠️")
def ok(msg: str):   st.success(msg, icon="✅")
def err(msg: str):  st.error(msg, icon="🟥")

# ============== GitHub API ==============
def gh_headers() -> Dict[str,str]:
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def gh_api_base() -> str:
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}"

def gh_get_contents(path: str) -> requests.Response:
    url = f"{gh_api_base()}/contents/{path}"
    return requests.get(url, headers=gh_headers(), params={"ref": GH_BRANCH}, timeout=30)

def gh_put_file(path: str, content: bytes, message: str) -> None:
    url = f"{gh_api_base()}/contents/{path}"
    sha = None
    r0 = gh_get_contents(path)
    if r0.status_code == 200:
        sha = r0.json().get("sha")
    payload = {"message": message, "content": base64.b64encode(content).decode(), "branch": GH_BRANCH}
    if sha: payload["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=payload, timeout=60)
    if r.status_code not in (200,201):
        raise RuntimeError(f"GitHub 寫入失敗：{r.status_code} {r.text}")

def gh_list_dirs(folder: str) -> List[str]:
    r = gh_get_contents(folder)
    if r.status_code != 200:
        return []
    return [it["name"] for it in r.json() if it.get("type")=="dir"]

def gh_list_excels(folder: str) -> List[str]:
    r = gh_get_contents(folder)
    if r.status_code != 200:
        return []
    paths = []
    for it in r.json():
        if it.get("type")=="file":
            name = it.get("name","")
            if name.lower().endswith((".xlsx",".xls")):
                paths.append(f"{folder}/{name}")
    return sorted(paths)

def gh_file_bytes(path: str) -> bytes:
    r = gh_get_contents(path)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub 讀檔失敗：{path} {r.status_code} {r.text}")
    data = r.json()
    return base64.b64decode(data["content"])

# ============== 欄位相容化 ==============
_fw = str.maketrans("ＡＢＣＤＥ１２３４５", "ABCDE12345")
_cn_to_L = {"一":"A","二":"B","三":"C","四":"D","五":"E","甲":"A","乙":"B","丙":"C","丁":"D","戊":"E"}

def _to_letter(h: str) -> Optional[str]:
    s = str(h).strip().replace(" ","").translate(_fw).upper()
    m = re.fullmatch(r"[（(]?([A-E])[)）]?", s)
    if m: return m.group(1)
    m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([A-E])", s)
    if m: return m.group(1)
    m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([1-5])", s)
    if m: return "ABCDE"[int(m.group(1))-1]
    m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([一二三四五甲乙丙丁戊])", s)
    if m: return _cn_to_L.get(m.group(1))
    return None

def normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    cmap = {
        "編號":"ID","題號":"ID","序號":"ID",
        "題目":"Question","題幹":"Question","題目內容":"Question",
        "答案":"Answer","正確答案":"Answer",
        "題型":"Type","類型":"Type",
        "解釋說明":"Explanation","解析":"Explanation","詳解":"Explanation","說明":"Explanation",
        "標籤":"Tag","章節":"Tag","科目":"Tag",
        "圖片":"Image","圖片連結":"Image",
    }
    df = df.rename(columns={c: cmap.get(c,c) for c in df.columns})

    # 選項欄位統一為 OptionA~E
    seen, ren = set(), {}
    for c in list(df.columns):
        L = _to_letter(c)
        if L and L not in seen:
            ren[c] = f"Option{L}"; seen.add(L)
    if ren: df = df.rename(columns=ren)

    option_cols = [f"Option{L}" for L in "ABCDE" if f"Option{L}" in df.columns]
    if len(option_cols) < 2:
        raise ValueError("題庫至少需要 2 個選項欄（OptionA/OptionB… 或 答案選項1/2… 等）")

    # 欄位補缺
    for c in ["ID","Question","Answer","Type","Explanation","Tag","Image",*option_cols]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str).str.strip()

    # 無 Answer → 從星號推導
    if "Answer" not in df.columns or df["Answer"].eq("").all():
        answers, types = [], []
        for i, r in df.iterrows():
            stars = []
            for idx, col in enumerate(option_cols):
                txt = str(r.get(col,"")).strip()
                if txt.startswith("*"):
                    stars.append(chr(ord("A")+idx))
                    df.at[i, col] = txt.lstrip("*").strip()
            if not stars:
                answers.append("")
                types.append("SC")
            else:
                answers.append("".join(stars))
                types.append("SC" if len(stars)==1 else "MC")
        df["Answer"] = answers
        if "Type" not in df.columns: df["Type"] = types

    for must in ["Question","Answer"]:
        if must not in df.columns:
            raise ValueError(f"缺少必要欄位：{must}")

    if "ID" not in df.columns: df["ID"] = ""
    if "Type" not in df.columns: df["Type"] = "SC"
    df["Type"]   = df["Type"].astype(str).str.upper().str.strip()
    df["Answer"] = df["Answer"].astype(str).str.upper().str.replace(" ","", regex=False)
    for c in ["Tag","Explanation","Image"]:
        if c not in df.columns: df[c] = ""

    # 至少兩個非空選項
    def _optcnt(row): return sum(1 for c in option_cols if str(row.get(c,"")).strip()!="")
    df = df[df["Answer"].str.len()>0].copy()
    df = df[df.apply(_optcnt, axis=1) >= 2].copy()
    return df, option_cols

# ============== 題庫載入（多檔/多分頁） ==============
def _engine_by_ext(path: str) -> Optional[str]:
    return "xlrd" if path.lower().endswith(".xls") else None

def _excel_file_obj(path: str, source: str) -> pd.ExcelFile:
    eng = _engine_by_ext(path)
    if source=="github":
        bio = io.BytesIO(gh_file_bytes(path))
        return pd.ExcelFile(bio, engine=eng) if eng else pd.ExcelFile(bio)
    else:
        return pd.ExcelFile(path, engine=eng) if eng else pd.ExcelFile(path)

def _read_sheet(path: str, sheet: str, source: str) -> pd.DataFrame:
    eng = _engine_by_ext(path)
    if source=="github":
        bio = io.BytesIO(gh_file_bytes(path))
        return pd.read_excel(bio, sheet_name=sheet, engine=eng) if eng else pd.read_excel(bio, sheet_name=sheet)
    else:
        return pd.read_excel(path, sheet_name=sheet, engine=eng) if eng else pd.read_excel(path, sheet_name=sheet)

def load_banks(files: List[str], selected_sheets: Dict[str,List[str]], use_sheet_as_tag: bool,
               auto_tag_from_id: bool, source: str) -> pd.DataFrame:
    rows = []
    for f in files:
        xls = _excel_file_obj(f, source)
        sheets = selected_sheets.get(f) or xls.sheet_names
        for sh in sheets:
            raw = _read_sheet(f, sh, source)
            df, _ = normalize_df(raw)

            # 補唯一 ID：<檔:頁:流水>
            stem = Path(f).stem
            prefix = f"{stem}:{sh}"
            df["ID"] = df["ID"].astype(str).str.strip()
            if df["ID"].eq("").any() or df["ID"].duplicated().any():
                df["ID"] = [f"{prefix}:{i+1}" for i in range(len(df))]
            else:
                df["ID"] = [f"{prefix}:{x}" for x in df["ID"]]

            # Tag 補強
            if "Tag" not in df.columns: df["Tag"] = ""
            if use_sheet_as_tag:
                mask = df["Tag"].astype(str).str.strip().eq("")
                df.loc[mask,"Tag"] = str(sh)
            if auto_tag_from_id:
                def head_token(x: str) -> str:
                    x=str(x).strip()
                    ps=re.split(r"[-_－—─:]",x,maxsplit=1)
                    return ps[0] if ps else x
                mask = df["Tag"].astype(str).str.strip().eq("")
                df.loc[mask,"Tag"] = df.loc[mask,"ID"].map(head_token)

            df["__file__"]  = f
            df["__sheet__"] = str(sh)
            rows.append(df)
    if not rows:
        raise RuntimeError("未載入任何題目。")
    out = pd.concat(rows, ignore_index=True)
    out["Tag"] = (out["Tag"].astype(str).str.replace("，", ",")).str.replace("；",";")
    out["Tag"] = out["Tag"].fillna("").astype(str).str.strip()
    return out

# ============== LLM（Gemini / 其餘保留作備援） ==============
def _ollama_ok(endpoint: str) -> bool:
    try:
        tags = endpoint.replace("/api/generate","/api/tags")
        return requests.get(tags, timeout=2).ok
    except Exception:
        return False

def pick_provider() -> str:
    prov = os.getenv("LLM_PROVIDER","").lower()
    if prov == "gemini" and os.getenv("GEMINI_API_KEY"): return "gemini"
    if prov == "openai" and os.getenv("OPENAI_API_KEY"): return "openai"
    if prov == "ollama" and os.getenv("OLLAMA_ENDPOINT"): return "ollama"
    # 自動偵測（有 key 就用）
    if os.getenv("GEMINI_API_KEY"): return "gemini"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if _ollama_ok(os.getenv("OLLAMA_ENDPOINT","http://127.0.0.1:11434/api/generate")): return "ollama"
    return "none"

@st.cache_data(show_spinner=False)
def llm_explain_cached(prompt: str, provider: str, model: str, endpoint: str) -> str:
    try:
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            mdl = model or os.getenv("GEMINI_MODEL","gemini-1.5-flash")
            gmodel = genai.GenerativeModel(mdl)
            resp = gmodel.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 400}
            )
            return (getattr(resp, "text", "") or "").strip()

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model or os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":"你是精準的保險學講師，使用繁體中文。"},
                          {"role":"user","content":prompt}],
                temperature=0.2, max_tokens=400
            )
            return (resp.choices[0].message.content or "").strip()

        elif provider == "ollama":
            payload = {"model": model or os.getenv("OLLAMA_MODEL","qwen2.5:3b-instruct"),
                       "prompt": prompt, "temperature": 0.2, "stream": False}
            r = requests.post(endpoint or os.getenv("OLLAMA_ENDPOINT","http://127.0.0.1:11434/api/generate"),
                              json=payload, timeout=120)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()

        else:
            return "（AI詳解未啟用）"

    except Exception as e:
        return f"（AI詳解失敗：{e}）"

def build_prompt(q: str, opts: Dict[str,str], correct: str, user: str, tag: str, ref: str) -> str:
    txt = "\n".join([f"{k}. {v}" for k,v in opts.items() if v])
    return (
        "請以繁體中文為考生產生針對本題的簡潔詳解：\n"
        "1) 一句話點出題幹關鍵概念；\n"
        "2) 說明正確選項為何正確；\n"
        "3) 指出考生選錯的關鍵誤解；\n"
        "4) 給 1 條易錯提醒。\n"
        f"章節：{tag or '（未標示）'}；來源：{ref}\n"
        f"題目：{q}\n選項：\n{txt}\n"
        f"正解：{correct}\n作答：{user or '未作答'}"
    )

# ============== UI：資料來源與管理 ==============
st.title("📘 模擬考與題庫練習")

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.sidebar:
    st.header("資料來源與管理")
    source_mode = "github" if gh_enabled() else "local"
    st.caption(f"來源模式：**{ 'GitHub / 題庫' if source_mode=='github' else '本機相對路徑 / 題庫' }**")

    # 管理員登入
    if not st.session_state.is_admin:
        if ADMIN_PASSWORD:
            with st.expander("管理員登入", True):
                pwd = st.text_input("管理密碼", type="password")
                if st.button("登入", use_container_width=True):
                    if pwd == ADMIN_PASSWORD:
                        st.session_state.is_admin = True
                        st.rerun()
                    else:
                        st.error("密碼錯誤")
        else:
            st.caption("（未設定 ADMIN_PASSWORD，可直接啟用管理模式）")
            if st.button("啟用管理模式"):
                st.session_state.is_admin = True
                st.rerun()
    else:
        st.success("管理模式啟用中")
        if st.button("登出管理模式"):
            st.session_state.is_admin = False
            st.rerun()

    # ===== 領域選擇 =====
    st.subheader("領域選擇")
    predefined_domains = ["人身","外幣","投資型"]
    if source_mode == "github":
        gh_dirs = gh_list_dirs(GH_FOLDER)
        domains = [d for d in predefined_domains if d in gh_dirs] or gh_dirs or ["(根目錄)"]
    else:
        local_dirs = [d.name for d in LOCAL_BANK_ROOT.iterdir() if d.is_dir()] if LOCAL_BANK_ROOT.exists() else []
        domains = [d for d in predefined_domains if d in local_dirs] or local_dirs or ["(根目錄)"]

    domain = st.selectbox("選擇領域", domains, key="domain_sel")

    # ===== 檔案多選 =====
    if source_mode == "github":
        base_path = GH_FOLDER if domain == "(根目錄)" else f"{GH_FOLDER}/{domain}"
        excel_paths = gh_list_excels(base_path)
        if st.session_state.is_admin:
            up = st.file_uploader("上傳 Excel 題庫到該領域", type=["xlsx","xls"], key="uploader_bank")
            if up is not None:
                try:
                    gh_put_file(f"{base_path}/{up.name}", up.read(), f"upload bank: {up.name}")
                    ok(f"已上傳 {up.name} 到 {base_path}")
                except Exception as e:
                    err(f"上傳失敗：{e}")
        selected_files = st.multiselect("選擇 Excel 檔（可多選）", options=excel_paths, key="files_sel")
    else:
        base_path = LOCAL_BANK_ROOT if domain == "(根目錄)" else LOCAL_BANK_ROOT / domain
        if not Path(base_path).exists():
            err(f"找不到資料夾：{base_path}")
            selected_files = []
        else:
            local_excels = [str(p) for p in Path(base_path).glob("*.xls*")]
            selected_files = st.multiselect("選擇 Excel 檔（可多選）", options=sorted(local_excels), key="files_sel")

    # ===== 分頁選擇 =====
    selected_sheets: Dict[str, List[str]] = {}
    with st.expander("分頁選擇（不選＝該檔全部分頁）", True):
        for f in selected_files:
            try:
                xls = _excel_file_obj(f, source_mode)
                selected_sheets[f] = st.multiselect(f"{Path(f).name} 的分頁",
                                                    options=xls.sheet_names,
                                                    key=f"__s_{f}")
            except Exception as e:
                err(f"讀取分頁失敗：{f}\n{e}")

    use_sheet_as_tag = st.checkbox("沒有 Tag 的題目，用分頁名作為 Tag", value=True)
    auto_tag_from_id = st.checkbox("從題號自動取章節（以 '-' 或 '_' 前段）", value=False)

# ============== 載入題庫（一次載入） ==============
@st.cache_data(show_spinner=True, ttl=300)
def _load_df(files: List[str], selected_sheets: Dict[str, List[str]], use_sheet_as_tag: bool, auto_tag_from_id: bool, src: str):
    return load_banks(files, selected_sheets, use_sheet_as_tag, auto_tag_from_id, src)

if not selected_files:
    info("請在左側選擇領域與題庫檔案。")
    st.stop()

try:
    df_bank = _load_df(selected_files, selected_sheets, use_sheet_as_tag, auto_tag_from_id,
                       "github" if gh_enabled() else "local")
    ok(f"已載入題目數：{len(df_bank)}")
except Exception as e:
    err(f"載入題庫失敗：{e}")
    st.stop()

# ============== 出題設定（固定後才抽題） ==============
with st.sidebar:
    st.header("出題設定")
    mode = st.radio("模式", ["練習","模考"], horizontal=True, index=0)
    # Tag 選單（去重）
    tags_series = df_bank["Tag"].fillna("").astype(str).str.strip()
    tags_series = tags_series[(tags_series!="") & (tags_series.str.lower()!="nan")]
    all_tags = sorted(tags_series.unique().tolist())
    picked_tags = st.multiselect("選擇章節/標籤（不選=全題庫）", options=all_tags, default=[])

    scope_df = df_bank if not picked_tags else df_bank[df_bank["Tag"].isin(picked_tags)]
    max_q = len(scope_df)
    qnum = st.number_input("題數", min_value=1, max_value=max(1,max_q), value=min(30,max_q), step=1)
    shuffle_opts = st.checkbox("選項亂序", True)
    shuffle_qs   = st.checkbox("題目亂序", True)
    show_img     = st.checkbox("顯示圖片欄（如有連結）", False)
    time_limit   = st.number_input("時間限制（分鐘；0=不限）", min_value=0, max_value=240, value=0, step=5)

    st.subheader("AI 詳解")
    use_ai   = st.checkbox("啟用 AI（僅錯題）", True)
    provider = pick_provider()
    st.caption(f"偵測供應者：**{provider}**")

    start = st.button("開始出題", use_container_width=True)
    reset = st.button("重新設定", use_container_width=True)

# ============== 抽題固定到 session_state ==============
def _build_pool_records(df: pd.DataFrame, n: int, shuffle_qs: bool, shuffle_opts: bool) -> List[dict]:
    if shuffle_qs:
        df = df.sample(frac=1.0, random_state=None).head(n).reset_index(drop=True)
    else:
        df = df.head(n).reset_index(drop=True)
    records = []
    for _, r in df.iterrows():
        opts = []
        for L in "ABCDE":
            v = str(r.get(f"Option{L}","")).strip()
            if v:
                opts.append((L,v))
        if shuffle_opts:
            import random
            random.shuffle(opts)
        records.append({
            "ID": r["ID"],
            "Question": r["Question"],
            "Answer": str(r["Answer"]).upper(),
            "Type": str(r["Type"]).upper(),
            "Tag": str(r.get("Tag","")),
            "Explanation": str(r.get("Explanation","")),
            "Image": str(r.get("Image","")),
            "__file__": str(r["__file__"]),
            "__sheet__": str(r["__sheet__"]),
            "Options": opts,
        })
    return records

if reset:
    for k in ["started","pool","answers","current_q","start_ts","settings","result_df"]:
        st.session_state.pop(k, None)
    st.rerun()

if start:
    st.session_state.settings = dict(
        mode=mode, shuffle_opts=shuffle_opts, shuffle_qs=shuffle_qs, show_img=show_img,
        time_limit=time_limit, use_ai=use_ai, provider=provider,
        # Gemini 參數（若你改名就用環境變數）
        gemini_model=os.getenv("GEMINI_MODEL","gemini-1.5-flash"),
        # 其餘供應者留作備援
        openai_model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
        ollama_model=os.getenv("OLLAMA_MODEL","qwen2.5:3b-instruct"),
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT","http://127.0.0.1:11434/api/generate"),
    )
    st.session_state.pool = _build_pool_records(scope_df, int(qnum), shuffle_qs, shuffle_opts)
    st.session_state.answers = {}
    st.session_state.current_q = 0
    if time_limit>0:
        st.session_state.start_ts = time.time()
    st.session_state.started = True
    st.rerun()

# ============== 顯示考試 ==============
if st.session_state.get("started"):
    s = st.session_state.settings
    pool: List[dict] = st.session_state.pool
    n = len(pool)

    # 倒數
    auto_submit = False
    if s["time_limit"]>0:
        elapsed = int(time.time() - st.session_state.get("start_ts", time.time()))
        remain = max(0, s["time_limit"]*60 - elapsed)
        mm, ss = divmod(remain, 60)
        st.warning(f"⏱️ 倒數 {mm:02d}:{ss:02d}")
        if remain==0:
            auto_submit = True

    idx = st.session_state.current_q
    r = pool[idx]
    st.markdown(f"### 第 {idx+1}/{n} 題")
    st.markdown(f"**{r['Question']}**")
    if s["show_img"] and r.get("Image"):
        st.caption(f"圖片：{r['Image']}")

    saved: List[str] = st.session_state.answers.get(idx, [])
    opt_keys = [k for k,_ in r["Options"]]
    opt_map  = {k:v for k,v in r["Options"]}

    # ---- 練習模式：即時判斷 ----
    if s["mode"] == "練習":
        if (r["Type"]=="MC" or len(r["Answer"])>1):
            if f"ans_{idx}" not in st.session_state:
                st.session_state[f"ans_{idx}"] = saved
            picked = st.multiselect("（複選）請選擇：", options=opt_keys,
                                    key=f"ans_{idx}",
                                    format_func=lambda k: opt_map[k],
                                    default=st.session_state[f"ans_{idx}"])
            st.session_state.answers[idx] = sorted(picked)
        else:
            if f"ans_{idx}" not in st.session_state:
                st.session_state[f"ans_{idx}"] = saved[0] if saved else None
            current_val = st.session_state[f"ans_{idx}"]
            radio_index = opt_keys.index(current_val) if current_val in opt_keys else None
            picked_value = st.radio("（單選）請選擇：", options=opt_keys,
                                    format_func=lambda k: opt_map[k],
                                    index=radio_index, key=f"radio_{idx}")
            st.session_state[f"ans_{idx}"] = picked_value if picked_value else None
            st.session_state.answers[idx] = [picked_value] if picked_value else []

        # 立刻判斷
        if st.session_state.answers.get(idx):
            user = "".join(sorted(st.session_state.answers[idx]))
            correct = "".join(sorted(r["Answer"]))
            ok_flag = (user==correct)
            if ok_flag:
                st.success(f"✅ 正確！（答案：{correct}）")
            else:
                st.error(f"❌ 錯誤（你的答案：{user}；正解：{correct}）")
            if r.get("Explanation"):
                st.info(f"原始詳解：{r['Explanation']}")
            if s["use_ai"] and not ok_flag and s["provider"]!="none":
                prompt = build_prompt(r["Question"], opt_map, correct, user, r.get("Tag",""),
                                      f"{Path(r['__file__']).name} / {r['__sheet__']}")
                ai = llm_explain_cached(prompt, s["provider"], s.get("gemini_model","gemini-1.5-flash"),
                                        s.get("ollama_endpoint",""))
                st.markdown(f"**AI詳解（僅錯題）**：{ai}")

    # ---- 模考模式：不即時顯示 ----
    else:
        if (r["Type"]=="MC" or len(r["Answer"])>1):
            if f"ans_{idx}" not in st.session_state:
                st.session_state[f"ans_{idx}"] = saved
            picked = st.multiselect("（複選）請選擇：", options=opt_keys,
                                    key=f"ans_{idx}",
                                    format_func=lambda k: opt_map[k],
                                    default=st.session_state[f"ans_{idx}"])
            st.session_state.answers[idx] = sorted(picked)
        else:
            if f"ans_{idx}" not in st.session_state:
                st.session_state[f"ans_{idx}"] = saved[0] if saved else None
            radio_index = opt_keys.index(st.session_state[f"ans_{idx}"]) if st.session_state[f"ans_{idx}"] in opt_keys else None
            picked_value = st.radio("（單選）請選擇：", options=opt_keys,
                                    format_func=lambda k: opt_map[k],
                                    index=radio_index, key=f"radio_{idx}")
            st.session_state[f"ans_{idx}"] = picked_value if picked_value else None
            st.session_state.answers[idx] = [picked_value] if picked_value else []

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        if st.button("⬅️ 上一題", disabled=(idx==0)):
            st.session_state.current_q -= 1
            st.rerun()
    with col2:
        if st.button("➡️ 下一題", disabled=(idx==n-1)):
            st.session_state.current_q += 1
            st.rerun()
    with col3:
        if st.button("回到第一題"):
            st.session_state.current_q = 0
            st.rerun()
    with col4:
        submit_clicked = st.button("交卷", type="primary")

    if auto_submit or submit_clicked:
        rows = []
        correct_cnt = 0
        for j, rr in enumerate(pool):
            user_letters = "".join(sorted(st.session_state.answers.get(j, [])))
            correct_letters = "".join(sorted(rr["Answer"]))
            ok_flag = (user_letters==correct_letters and user_letters!="")
            if ok_flag: correct_cnt += 1

            ai_exp = ""
            if s["use_ai"] and not ok_flag and s["provider"]!="none":
                o_map = {k:v for k,v in rr["Options"]}
                prompt = build_prompt(rr["Question"], o_map, correct_letters, user_letters, rr.get("Tag",""),
                                      f"{Path(rr['__file__']).name} / {rr['__sheet__']}")
                ai_exp = llm_explain_cached(prompt, s["provider"], s.get("gemini_model","gemini-1.5-flash"),
                                            s.get("ollama_endpoint",""))

            rows.append({
                "ID": rr["ID"], "Tag": rr.get("Tag",""), "Question": rr["Question"],
                "YourAnswer": user_letters, "Correct": correct_letters,
                "Result": "O" if ok_flag else "X",
                "Explanation": rr.get("Explanation",""),
                "AI_Explanation": ai_exp if (s["use_ai"] and not ok_flag) else "",
                "File": Path(rr["__file__"]).name, "Sheet": rr["__sheet__"],
            })

        score = round(correct_cnt / len(pool) * 100, 2)
        st.subheader(f"🧾 成績：{correct_cnt}/{len(pool)}（{score} 分）")
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True)

        csv = df_res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載成績（CSV）",
                           data=csv,
                           file_name=f"exam_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

        if st.button("再考一次（同樣設定）"):
            st.session_state.pop("started", None)
            st.rerun()

else:
    info("請在左側完成『領域／檔案／分頁』與『出題設定』後，按下『開始出題』。")

st.caption("來源模式：" + ("GitHub / 題庫" if gh_enabled() else f"本機 / {LOCAL_BANK_ROOT}"))
