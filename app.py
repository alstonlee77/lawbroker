# app.py
# -*- coding: utf-8 -*-
import os
import io
import re
import time
import json
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests
import streamlit as st


# =========================
# 基本設定
# =========================
st.set_page_config(page_title="模擬考與題庫練習", layout="wide")
st.title("📘 模擬考與題庫練習")

# 你的三大領域根目錄（人身 / 外幣 / 投資型 都在這底下）
BASE_ROOT = Path("/Users/lch/lawbroker/考照AI")   # ← 路徑不同請改這行

# 僅用於排序置頂；實際以目錄掃描為準
DEFAULT_DOMAIN_NAMES = ["人身", "外幣", "投資型"]
VALID_EXTS = (".xlsx", ".xls")  # .xls 需 xlrd<2.0；或把檔案另存為 .xlsx


# =========================
# 管理者判斷（不依賴 secrets）
# =========================
def resolve_admin() -> bool:
    try:
        qp = st.query_params
        admin_val = qp.get("admin", None)
        if admin_val is not None:
            if isinstance(admin_val, list):
                admin_val = admin_val[0]
            if str(admin_val) == "1":
                return True
    except Exception:
        pass
    if os.getenv("ADMIN", "0") == "1" or os.getenv("STREAMLIT_ADMIN", "0") == "1":
        return True
    try:
        return str(st.secrets.get("ADMIN", "0")) == "1"
    except Exception:
        return False

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass

IS_ADMIN = resolve_admin()


# =========================
# AI 詳解：只在錯題時觸發（預設走本機 Ollama）
# =========================
AI_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()    # openai / ollama
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434/api/generate")

def _hash_key(payload: dict) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def llm_explain(question: str,
                options: dict,
                correct_letters: List[str],
                user_letters: List[str],
                tag: str = "",
                domain: str = "",
                file_name: str = "",
                sheet_name: str = "") -> str:
    """
    呼叫語言模型產生中文詳解（簡潔、可教學；不暴露思維過程）。
    僅在錯題時被呼叫。
    """
    prompt = f"""
你是一位保險/金融考試講解老師。請用繁體中文，給出「精簡且可教學」的解析。
限制：
- 只以題目與選項為依據；不確定時請指出資料不足。
- 結構 3 段：1) 為何正解是 {''.join(correct_letters)}；2) 本題常見誤解與排除；3) 一句帶走重點。
- 嚴禁捏造未在題幹與選項出現的專有名詞或數值。
- 120–220 字。

[題目領域] {domain}
[來源] 檔案：{file_name} / 分頁：{sheet_name} / 標籤：{tag}
[題幹]
{question}

[選項]
{options}

[正確答案] {''.join(correct_letters)}
[使用者作答] {''.join(user_letters) if user_letters else '（未作答）'}
"""
    try:
        if AI_PROVIDER == "openai":
            try:
                from openai import OpenAI
            except Exception:
                return "（AI詳解無法啟動：請先安裝 openai 套件或設定 OPENAI_API_KEY）"
            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You provide concise explanations without revealing chain-of-thought. Output only the final explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return (resp.choices[0].message.content or "").strip()
        else:
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": 0.2, "stream": False}
            r = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
    except Exception as e:
        return f"（AI詳解產生失敗：{e}）"

def get_ai_explanation(qrow: pd.Series,
                       user_letters: List[str],
                       option_cols: List[str]) -> str:
    """
    附帶快取的 AI 詳解。只在錯題時被上層呼叫。
    """
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    opts = {}
    for idx, col in enumerate(option_cols):
        txt = str(qrow.get(col, "")).strip()
        if not txt:
            continue
        letter = chr(ord('A') + idx)
        opts[letter] = txt

    payload = {
        "id": str(qrow.get("ID", "")),
        "question": str(qrow.get("Question", "")),
        "options": opts,
        "gold": list(str(qrow.get("Answer", "")).strip()),
        "user": user_letters,
        "provider": AI_PROVIDER,
        "model": OPENAI_MODEL if AI_PROVIDER == "openai" else OLLAMA_MODEL
    }
    key = _hash_key(payload)
    if key in st.session_state.llm_cache:
        return st.session_state.llm_cache[key]

    text = llm_explain(
        question=payload["question"],
        options=payload["options"],
        correct_letters=payload["gold"],
        user_letters=payload["user"],
        tag=str(qrow.get("Tag", "")),
        domain=str(st.session_state.get("current_domain", "")),
        file_name=os.path.basename(str(qrow.get("__file__", ""))),
        sheet_name=str(qrow.get("__sheet__", "")),
    )
    st.session_state.llm_cache[key] = text
    return text


# =========================
# 題庫載入：單表正規化
# =========================
def _normalize_bank_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """把單一工作表的題庫欄位正規化（擴充：答案選項1..5 / A..E / 甲乙丙丁 / 全形數字 等）。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # ---- 常見欄位對映 ----
    col_map = {
        "編號": "ID","題目編號":"ID", "題目": "Question", "題幹": "Question", "題目內容": "Question", "題目敘述": "Question",
        "答案": "Answer", "正確答案": "Answer", "標準答案": "Answer",
        "題型": "Type", "類型": "Type",
        "解釋說明": "Explanation", "解析": "Explanation", "詳解": "Explanation", "說明": "Explanation",
        "標籤": "Tag", "章節": "Tag", "科目": "Tag",
        "圖片": "Image", "圖片連結": "Image",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    # ---- 選項欄偵測與 rename 成 OptionA..OptionE ----
    fw = str.maketrans("ＡＢＣＤＥ１２３４５", "ABCDE12345")
    cn_to_letter = {"一": "A", "二": "B", "三": "C", "四": "D", "五": "E",
                    "甲": "A", "乙": "B", "丙": "C", "丁": "D", "戊": "E"}

    def to_letter_from_header(h: str) -> Optional[str]:
        # 例：答案選項1、選項1、選1、項1、1、答案選項A、A、(A)、（A）
        s = str(h).strip().replace(" ", "")
        s = s.translate(fw).upper()

        # 直接 A~E 或 (A) / （A）
        m = re.fullmatch(r"[（(]?([A-E])[)）]?", s)
        if m:
            return m.group(1)

        # 帶前綴：答案選項A / 選項A / 選A / 項A
        m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([A-E])", s)
        if m:
            return m.group(1)

        # 1~5 對應 A~E（含全形）
        m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([1-5])", s)
        if m:
            return "ABCDE"[int(m.group(1)) - 1]

        # 中文數字/甲乙丙丁戊
        m = re.fullmatch(r"(?:答案)?(?:選項|選|項)?([一二三四五甲乙丙丁戊])", s)
        if m:
            return cn_to_letter.get(m.group(1))
        return None

    rename_map: Dict[str, str] = {}
    seen_letters: set = set()
    for col in list(df.columns):
        letter = to_letter_from_header(col)
        if letter and letter not in seen_letters:
            rename_map[col] = f"Option{letter}"
            seen_letters.add(letter)

    if rename_map:
        df = df.rename(columns=rename_map)

    # 組出最後的選項欄位清單（照 A~E 排序，僅保留存在的）
    option_cols = [f"Option{L}" for L in "ABCDE" if f"Option{L}" in df.columns]
    if len(option_cols) < 2:
        raise ValueError("題庫至少需要 2 個選項欄（例如 OptionA/OptionB 或 答案選項1/答案選項2）。")

    # ---- 基本欄位補空 + 去空白 ----
    for c in ["ID", "Question", "Answer", "Type", "Explanation", "Tag", "Image", *option_cols]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    # ---- 若沒有 Answer 或 Answer 全空：以「*」在選項內推導答案 ----
    if "Answer" not in df.columns or df["Answer"].eq("").all():
        answers, types = [], []
        for i, r in df.iterrows():
            starred_letters: List[str] = []
            for idx, col in enumerate(option_cols):
                txt = str(r[col]).strip()
                if txt.startswith("*"):
                    starred_letters.append(chr(ord("A") + idx))
                    df.at[i, col] = txt.lstrip("*").strip()
            if len(starred_letters) == 0:
                answers.append("")
                types.append("SC")
            elif len(starred_letters) == 1:
                answers.append("".join(starred_letters))
                types.append("SC")
            else:
                answers.append("".join(starred_letters))
                types.append("MC")
        df["Answer"] = answers
        if "Type" not in df.columns:
            df["Type"] = types

    # ---- 必要欄位檢查與正規化 ----
    for col in ["ID", "Question", "Answer"]:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位：{col}")

    if "Type" not in df.columns:
        df["Type"] = "SC"
    df["Type"] = df["Type"].astype(str).str.upper().str.strip()
    df["Answer"] = (df["Answer"].astype(str)
                               .str.upper()
                               .str.replace(" ", "", regex=False))

    for c in ["Tag", "Explanation", "Image"]:
        if c not in df.columns:
            df[c] = ""

    # 至少兩個非空選項
    def option_count(row) -> int:
        return sum(1 for c in option_cols if str(row.get(c, "")).strip() != "")
    df = df[df["Answer"].str.len() > 0].copy()
    df = df[df.apply(option_count, axis=1) >= 2].copy()

    return df, option_cols


@st.cache_data(show_spinner=False)
def load_bank_multi_files(
    file_sheet_pairs: List[Tuple[str, str]],
    autofill_tag_with_sheet: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    讀取「多個 Excel + 多個分頁」並合併，回傳 (合併後題庫, 選項欄位清單)。
    會附上 __file__ / __sheet__ 欄位以追蹤來源。
    """
    if not file_sheet_pairs:
        raise ValueError("未選取任何檔案/分頁。")

    merged_list: List[pd.DataFrame] = []
    first_opt_cols: Optional[List[str]] = None

    from collections import defaultdict
    group: Dict[str, List[str]] = defaultdict(list)
    for f, sh in file_sheet_pairs:
        group[f].append(sh)

    for fpath, sheets in group.items():
        try:
            xls = pd.ExcelFile(fpath)
        except Exception as e:
            raise RuntimeError(f"無法開啟檔案：{fpath}\n{e}\n（.xls 請安裝 xlrd<2.0 或改存 .xlsx）")

        sheets_to_read = sheets or xls.sheet_names
        for sh in sheets_to_read:
            try:
                df_raw = pd.read_excel(xls, sheet_name=sh)
            except Exception as e:
                raise RuntimeError(f"讀取分頁失敗：{fpath} :: {sh}\n{e}")

            df_norm, opt_cols = _normalize_bank_df(df_raw)
            if autofill_tag_with_sheet:
                df_norm["Tag"] = df_norm["Tag"].apply(lambda v: v if str(v).strip() else str(sh))
            df_norm["__file__"] = str(fpath)
            df_norm["__sheet__"] = str(sh)
            merged_list.append(df_norm)
            if first_opt_cols is None:
                first_opt_cols = opt_cols

    if not merged_list:
        raise ValueError("選取的檔案/分頁為空。")

    merged = pd.concat(merged_list, ignore_index=True)
    return merged, first_opt_cols or []


# =========================
# 出題/評分輔助
# =========================
def sample_paper(df: pd.DataFrame, n: int, tags: List[str] | None = None) -> pd.DataFrame:
    pool = df.copy()
    if tags and "Tag" in pool.columns:
        pool = pool[pool["Tag"].isin(tags)]
    if len(pool) == 0:
        raise ValueError("沒有符合條件的題目。")
    n = min(n, len(pool))
    return pool.sample(n=n, random_state=random.randint(1, 10_000)).reset_index(drop=True)

def build_display_options(row: pd.Series, option_cols: List[str], shuffle: bool = True) -> List[str]:
    items = []
    for idx, col in enumerate(option_cols):
        txt = str(row.get(col, "")).strip()
        if not txt:
            continue
        letter = chr(ord('A') + idx)
        items.append(f"{letter}. {txt}")
    if shuffle:
        random.shuffle(items)
    return items

def parse_choice_to_letters(picked) -> List[str]:
    if isinstance(picked, str):
        picked = [picked]
    letters = []
    for it in picked or []:
        head = str(it).split(".", 1)[0].strip().upper()
        if head and head[0].isalpha():
            letters.append(head[0])
    return letters

def grade(paper: pd.DataFrame, answers: Dict[int, List[str]]) -> tuple[pd.DataFrame, int]:
    rows, score = [], 0
    for i, r in paper.iterrows():
        gold = sorted(list(set(str(r["Answer"]))))
        pred = sorted(list(set(answers.get(i, []))))
        correct = (gold == pred)
        score += int(correct)
        rows.append({
            "ID": r["ID"],
            "Tag": r.get("Tag", ""),
            "Question": r["Question"],
            "Your Answer": "".join(pred),
            "Correct Answer": "".join(gold),
            "Correct": "✅" if correct else "❌",
            "Explanation": r.get("Explanation", ""),
            "File": r.get("__file__", ""),
            "Sheet": r.get("__sheet__", ""),
        })
    return pd.DataFrame(rows), score


# =========================
# 側欄：領域 → 檔案 → 分頁 → 出題設定（AI 只在錯題時）
# =========================
with st.sidebar:
    st.subheader("📂 領域選擇")
    if not BASE_ROOT.exists():
        st.error(f"找不到根目錄：{BASE_ROOT}")
        st.stop()

    subdirs = [d for d in BASE_ROOT.iterdir() if d.is_dir()]
    sorted_subdirs = sorted(
        subdirs, key=lambda p: (DEFAULT_DOMAIN_NAMES.index(p.name) if p.name in DEFAULT_DOMAIN_NAMES else 999, p.name)
    )
    domain_names = [p.name for p in sorted_subdirs] or DEFAULT_DOMAIN_NAMES
    domain = st.selectbox("選擇領域", options=domain_names, index=0 if domain_names else 0)
    st.session_state.current_domain = domain

    domain_path = BASE_ROOT / domain
    excel_files = sorted([str(p) for p in domain_path.glob("**/*") if p.suffix.lower() in VALID_EXTS])

    st.subheader("📄 檔案選擇")
    files_selected = st.multiselect("選擇一個或多個 Excel 檔", options=excel_files)

    # 掃描分頁
    sheet_options: List[str] = []
    file_sheet_map: Dict[str, List[str]] = {}
    for f in files_selected:
        try:
            xls = pd.ExcelFile(f)
            file_sheet_map[f] = xls.sheet_names
            sheet_options.extend([f"{Path(f).name} :: {sh}" for sh in xls.sheet_names])
        except Exception as e:
            st.warning(f"無法讀取：{f}\n{e}")

    st.subheader("📑 分頁選擇")
    picked_pairs_display = st.multiselect(
        "選擇要載入的分頁（不選＝把所選檔案的所有分頁都載入）",
        options=sheet_options
    )

    def parse_pairs(selected_display: List[str]) -> List[Tuple[str, str]]:
        if not selected_display:
            pairs = []
            for f, sheets in file_sheet_map.items():
                for sh in sheets:
                    pairs.append((f, sh))
            return pairs
        pairs = []
        name_to_file = {Path(f).name: f for f in file_sheet_map.keys()}
        for item in selected_display:
            try:
                fname, sh = [s.strip() for s in item.split("::", 1)]
                fpath = name_to_file.get(fname)
                if fpath:
                    pairs.append((fpath, sh))
            except Exception:
                continue
        return pairs

    file_sheet_pairs = parse_pairs(picked_pairs_display)

    autofill_tag = st.checkbox("沒有 Tag 的題目，用分頁名作為 Tag", value=False)

    st.subheader("🧰 出題設定")
    mode = st.radio("模式", ["練習", "模考"], horizontal=True)

    tag_placeholder = st.empty()  # 之後填 options
    num_q = st.number_input("題數", min_value=1, max_value=500, value=30)
    time_min = st.number_input("限時（分，0=不限時）", min_value=0, max_value=240, value=0)
    shuffle_options = st.checkbox("亂序顯示選項", value=True)

    st.subheader("🧠 AI 詳解（僅錯題）")
    ai_enable = st.checkbox("啟用 AI 產生詳解（只在錯題時）", value=True)

    start_btn = st.button("▶ 開始/重抽")


# =========================
# 載入題庫（跨檔案 + 跨分頁）
# =========================
df_bank, option_cols = None, []
if files_selected:
    try:
        df_bank, option_cols = load_bank_multi_files(file_sheet_pairs, autofill_tag_with_sheet=autofill_tag)
    except Exception as e:
        st.error(f"載入題庫失敗：\n{e}")
        st.stop()
else:
    st.info("請先在左側選擇領域與檔案（可多選），再選分頁。")
    st.stop()

# Tag 選單（防呆）
with st.sidebar:
    if "Tag" in df_bank.columns:
        all_tags = sorted(
            t for t in df_bank["Tag"].dropna().astype(str).map(str.strip).tolist()
            if t and t.lower() != "nan"
        )
    else:
        all_tags = []
    picked_tags = tag_placeholder.multiselect("選擇章節/標籤（可多選；不選=全題庫）", options=all_tags, default=[])


# =========================
# 初始化/開始一場考試
# =========================
def start_new_session(df: pd.DataFrame):
    try:
        paper = sample_paper(df, int(st.session_state.num_q), st.session_state.picked_tags)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.session_state.paper = paper
    st.session_state.start_ts = time.time()
    st.session_state.time_limit = int(st.session_state.time_min) * 60
    st.session_state.session_id = str(int(st.session_state.start_ts))
    st.session_state.answers = {}
    st.session_state.display_map = {}
    st.session_state.submitted = False

st.session_state.num_q = int(num_q)
st.session_state.time_min = int(time_min)
st.session_state.picked_tags = picked_tags

if start_btn or "paper" not in st.session_state:
    start_new_session(df_bank)
    if start_btn:
        safe_rerun()

paper: pd.DataFrame = st.session_state.paper
session_id: str = st.session_state.session_id


# =========================
# 倒數與時間控制
# =========================
elapsed = int(time.time() - st.session_state.start_ts)
remain = max(0, (st.session_state.time_limit or 0) - elapsed)
time_up = (st.session_state.time_limit and remain == 0)
if st.session_state.time_limit:
    st.info(f"⏱️ 剩餘時間：{remain // 60:02d}:{remain % 60:02d}")
if time_up and not st.session_state.submitted:
    st.warning("時間到！系統已自動交卷。")
    st.session_state.submitted = True


# =========================
# 題目渲染（練習模式錯題即時 AI）
# =========================
for idx, row in paper.iterrows():
    with st.container(border=True):
        st.markdown(f"**Q{idx+1}. {row['Question']}**")
        src_file = Path(str(row.get('__file__', ''))).name
        src_sheet = str(row.get('__sheet__', ''))
        st.caption(f"來源：{src_file} / {src_sheet}")

        img_url = str(row.get("Image", "")).strip()
        if img_url:
            st.image(img_url, use_column_width=True)

        display = st.session_state.display_map.get(idx)
        if display is None:
            display = build_display_options(row, option_cols, shuffle=shuffle_options)
            st.session_state.display_map[idx] = display

        qtype = (str(row.get("Type", "SC")).upper().strip() or "SC")
        key_prefix = f"{session_id}_q_{idx}"
        disabled = (st.session_state.submitted or time_up)

        if qtype == "MC":
            picked = st.multiselect("（複選）", options=display, key=key_prefix, disabled=disabled)
            letters = parse_choice_to_letters(picked)
        else:
            choice = st.radio("（單選）", options=display, key=key_prefix, disabled=disabled)
            letters = parse_choice_to_letters(choice)

        st.session_state.answers[idx] = letters

        # 練習模式：僅錯題時出 AI 詳解
        if mode == "練習" and not disabled and letters is not None:
            gold = sorted(list(set(str(row["Answer"]))))
            pred = sorted(list(set(letters)))
            if pred == gold:
                st.success(f"✅ 正確（答案：{''.join(gold)}）")
                expl = str(row.get("Explanation", "")).strip()
                if expl:
                    st.info(f"詳解：{expl}")
            else:
                st.error(f"❌ 錯誤（正解：{''.join(gold)}）")
                expl = str(row.get("Explanation", "")).strip()
                if expl:
                    st.info(f"詳解：{expl}")
                if ai_enable:
                    with st.spinner("AI 正在為錯題產生詳解…"):
                        ai_text = get_ai_explanation(row, pred, option_cols)
                    if ai_text:
                        st.info(f"🧠 AI詳解（錯題）：\n\n{ai_text}")

st.divider()


# =========================
# 交卷與結果（僅對錯題產生 AI 欄）
# =========================
submit_clicked = st.button("📥 交卷並看成績", use_container_width=True, disabled=st.session_state.submitted)
if submit_clicked:
    st.session_state.submitted = True

if st.session_state.submitted:
    result_df, score = grade(paper, st.session_state.answers)
    st.subheader("成績")
    st.metric("得分", f"{score} / {len(paper)}")

    if ai_enable:
        ai_texts = []
        for i, r in paper.iterrows():
            gold = sorted(list(set(str(r["Answer"]))))
            pred = sorted(list(set(st.session_state.answers.get(i, []))))
            if pred != gold:  # 只針對錯題
                ai_texts.append(get_ai_explanation(r, pred, option_cols))
            else:
                ai_texts.append("")
        result_df["AI Explanation (Wrong Only)"] = ai_texts

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    @st.cache_data
    def to_csv(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ 下載作答明細（CSV）",
        data=to_csv(result_df),
        file_name=f"exam_result_{session_id}.csv",
        mime="text/csv"
    )

    st.caption("提示：調整左側設定並按「開始/重抽」即可開新場次。")
else:
    st.caption("填完答案後按下方按鈕交卷；或到時會自動交卷。")
