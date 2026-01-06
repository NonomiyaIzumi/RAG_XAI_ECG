import re, json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# ------------ Cấu hình ------------
JSONL_IN = Path("New_method_outputs.jsonl")       # sửa path nếu cần
CSV_IN   = Path("ptbxl_database.csv")                       # sửa path nếu cần
CSV_OUT_CORRECT   = Path("match_new_pipeline_gemini_correct_gemini_metrics.csv")
CSV_OUT_INCORRECT = Path("match_new_pipeline_gemini_incorrect_gemini_metrics.csv")
CSV_OUT_SKIPPED   = Path("match_new_pipeline_gemini_skip_gemini_metrics.csv")  # (tùy chọn) xem các index bị bỏ qua

# Gắn trực tiếp API key của Onii-chan ở đây
GEMINI_API_KEY = "GEMINI_API_KEY"   # <--- điền key vào
USE_GEMINI = True        # True = gọi Gemini (dịch + trọng tài), False = chỉ heuristic offline
GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_GEMINI_CALLS = 2000  # Giới hạn tổng số lần gọi Gemini (dịch + chấm)
TRANSLATE_IMPRESSION_TO_DE = True
# ----------------------------------

# Stopwords EN + DE (ngắn gọn, đủ dùng cho so khớp)
STOPWORDS = set("""
a an the and or of to in on at for with without into from by over under between among about above below across
is are was were be been being this that these those it its as than then so such very can could should would
there their them they he she we you your our his her my mine yours ours theirs not no nor only also than like
im am pm

und oder der die das ein eine einem einen einer eines mit ohne über unter zwischen bei vom zum zur im am
ist sind war waren wird werden nicht kein keine nur auch wie so für gegen an auf aus nach vor seit bis dass daß
""".split())

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9äöüß\s]+", " ", s)  # giữ umlaut/ß cho DE
    s = re.sub(r"\s+", " ", s).strip()
    return s

def content_tokens(s: str) -> List[str]:
    toks = normalize(s).split()
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]

def jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b: return 1.0
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def overlap_min_ratio(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / float(min(len(A), len(B)))

# --------- Nhận diện report cần bỏ qua ----------
MOJIBAKE_FLAGS = ("Ã", "â", "�")  # các dấu hiệu lỗi mã hóa phổ biến
TRACE_PAT = re.compile(r"\btrace\s+only\s+requested\b", re.I)

def looks_unreadable(s: str) -> bool:
    if not s or not s.strip():
        return True
    if TRACE_PAT.search(s or ""):
        return True
    # Rất nhiều ký tự lỗi/mã hóa kém → bỏ qua
    if any(flag in s for flag in MOJIBAKE_FLAGS):
        return True
    # Quá nhiều dấu hỏi → nghi ngờ không đọc được
    if s.count("?") >= 3:
        return True
    # Quá ít chữ cái → chuỗi không có nội dung hữu ích
    letters = sum(ch.isalpha() for ch in s)
    return letters < 5

# ---------- Chọn cột report ----------
def pick_report_column(df: pd.DataFrame) -> str:
    if "report_de" in df.columns: return "report_de"
    if "report" in df.columns:    return "report"
    if "report_en" in df.columns: return "report_en"
    raise KeyError("Không tìm thấy cột report_de/report/report_en trong CSV.")

# ---------- Gemini utils ----------
_gem_client = None
def _get_gem_client():
    global _gem_client
    if _gem_client is None:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gem_client = genai.GenerativeModel(GEMINI_MODEL)
    return _gem_client

def translate_to_de(text: str) -> Optional[str]:
    """Dịch sang tiếng Đức. Trả None nếu lỗi/không có key."""
    if not GEMINI_API_KEY or not USE_GEMINI or not TRANSLATE_IMPRESSION_TO_DE:
        return None
    try:
        client = _get_gem_client()
        prompt = (
            "You are a professional medical translator specialized in ECG terminology.\n"
            "Translate the following IMPRESSION into precise clinical German.\n"
            "- Preserve all ECG terms (ST, QT, QRS, RBBB/LBBB, Axis, AV-block, etc.).\n"
            "- Do NOT add new findings.\n"
            "- Do NOT remove uncertain, partial, or vague findings.\n"
            "- Do NOT reinterpret the meaning.\n"
            "- Output ONLY the German translation, no quotes, no justification.\n\n"
            f"IMPRESSION:\n{text}\n"
        )

        resp = client.generate_content(prompt)
        out = (resp.text or "").strip()
        return out or None
    except Exception:
        return None

def judge_with_gemini(impression_de: str, report_de: str) -> Optional[bool]:
    if not GEMINI_API_KEY or not USE_GEMINI:
        return None
    try:
        client = _get_gem_client()
        prompt = (
            "You are an impartial and strict medical report adjudicator.\n"
            "Both texts are in GERMAN.\n"
            "Your task: Determine whether the IMPRESSION is *clinically consistent* with the REPORT.\n"
            "\nStrict rules:\n"
            "- ONLY evaluate clinical agreement.\n"
            "- Ignore differences in writing style, formatting, synonyms.\n"
            "- If IMPRESSION includes findings that also appear in the REPORT → YES.\n"
            "- If IMPRESSION contradicts REPORT → NO.\n"
            "- If REPORT contains additional findings *not mentioned* in the IMPRESSION → still YES.\n"
            "- If IMPRESSION states something not supported OR contradicted by REPORT → NO.\n"
            "- Do NOT interpret, assume, infer, or guess missing information.\n"
            "- If uncertain, choose NO.\n"
            "\nReturn strictly:\n"
            "- 'YES' → matches clinically\n"
            "- 'NO' → does not match\n"
            "No explanation.\n\n"
            f"IMPRESSION (DE): {impression_de}\n"
            f"REPORT (DE): {report_de}\n"
        )

        resp = client.generate_content(prompt)
        text = (resp.text or "").strip().upper()
        if text.startswith("YES") or ("YES" in text and "NO" not in text): return True
        if text.startswith("NO")  or ("NO" in text and "YES" not in text): return False
    except Exception:
        return None
    return None

# ---------- Heuristic offline (dùng DE + EN thuật ngữ ECG) ----------
ECG_TERMS_EN = [
    "sinus","bradycardia","tachycardia","infarction","ischemia",
    "st depression","st elevation","lbbb","rbbb","block","hypertrophy",
    "axis","arrhythmia","atrial","ventricular","pacemaker",
    "low voltage","q wave","t wave","sttc","cd","mi","norm","normal"
]
ECG_TERMS_DE = [
    "sinus","bradykardie","tachykardie","infarkt","ischämie","ischemie",
    "st-senkung","st-hebung","lbbb","rbbb","block","hypertrophie",
    "achse","arrhythmie","vorhof","kammer","schrittmacher",
    "niedrige spannung","q-zacke","t-welle","st-strecke","av-block","schenkelblock","normal"
]

def judge_offline(impression_de: str, report_de: str) -> bool:
    if not impression_de or not report_de: return False
    imp_norm = normalize(impression_de); rep_norm = normalize(report_de)
    if not imp_norm or not rep_norm: return False
    # Bao hàm chuỗi đơn giản
    if imp_norm in rep_norm or rep_norm in imp_norm: return True
    # Token overlap
    toks_imp = content_tokens(impression_de); toks_rep = content_tokens(report_de)
    if jaccard(toks_imp, toks_rep) >= 0.12: return True
    if overlap_min_ratio(toks_imp, toks_rep) >= 0.35: return True
    # Từ khóa ECG (DE + EN) xuất hiện ở cả hai bên
    if (any(term in imp_norm for term in ECG_TERMS_DE + ECG_TERMS_EN) and
        any(term in rep_norm for term in ECG_TERMS_DE + ECG_TERMS_EN)):
        return True
    return False

# ---------- Đọc dữ liệu ----------
df_reports = pd.read_csv(CSV_IN)
rep_col = pick_report_column(df_reports)
pos_to_report: Dict[int, str] = df_reports[rep_col].fillna("").astype(str).to_dict()

# đọc JSONL
records: List[Dict[str, Any]] = []
with JSONL_IN.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        idx = obj.get("index", None)
        go  = obj.get("gemini_output", {}) or {}
        impression = go.get("impression", "")
        if isinstance(impression, list):
            impression = "; ".join(str(x) for x in impression)
        impression = str(impression).strip()
        report_txt = pos_to_report.get(idx, "")
        records.append({"index": idx, "impression_raw": impression, "report_de": report_txt})

# ---------- Dịch impression -> DE (có cache) ----------
translation_cache: Dict[str, str] = {}
def get_impression_de(imp: str, budget: Dict[str, int]) -> str:
    base = (imp or "").strip()
    if not base:
        return ""
    if base in translation_cache:
        return translation_cache[base]
    out = translate_to_de(base)
    if out:
        translation_cache[base] = out
        budget["calls"] += 1
        return out
    # fallback: nếu không dịch được thì dùng nguyên văn (vẫn so khớp heuristic)
    translation_cache[base] = base
    return base

# ---------- Chấm điểm ----------
correct_rows, incorrect_rows, skipped_rows = [], [], []
gemini_budget = {"calls": 0}
gemini_used = False

for r in records:
    idx = r["index"]
    rep = r["report_de"]
    # Skip nếu report rỗng/trace/lỗi phông
    if looks_unreadable(rep):
        skipped_rows.append({"index": idx, "impression": r["impression_raw"], "report": rep, "reason": "unreadable/trace/empty"})
        continue

    # Dịch impression sang DE trước khi so sánh
    imp_de = get_impression_de(r["impression_raw"], gemini_budget)

    # Gọi Gemini làm trọng tài (nếu còn ngân sách)
    verdict = None
    if USE_GEMINI and gemini_budget["calls"] < MAX_GEMINI_CALLS:
        v = judge_with_gemini(imp_de, rep)
        if v is not None:
            verdict = v
            gemini_used = True
            gemini_budget["calls"] += 1

    # Heuristic offline nếu Gemini không trả lời
    if verdict is None:
        verdict = judge_offline(imp_de, rep)

    (correct_rows if verdict else incorrect_rows).append(
        {"index": idx, "impression": imp_de, "report": rep}
    )

# ---------- Xuất file ----------
pd.DataFrame(correct_rows).to_csv(CSV_OUT_CORRECT, index=False, encoding="utf-8")
pd.DataFrame(incorrect_rows).to_csv(CSV_OUT_INCORRECT, index=False, encoding="utf-8")
pd.DataFrame(skipped_rows).to_csv(CSV_OUT_SKIPPED, index=False, encoding="utf-8")

print(f"✅ Đã lưu: {CSV_OUT_CORRECT} ({len(correct_rows)} dòng)")
print(f"✅ Đã lưu: {CSV_OUT_INCORRECT} ({len(incorrect_rows)} dòng)")
print(f"⏭️  Bỏ qua (không tính): {CSV_OUT_SKIPPED} ({len(skipped_rows)} dòng)")
print(f"Gemini được sử dụng: {gemini_used} | Số lần gọi (dịch + chấm): {gemini_budget['calls']}")
print(f"Report column dùng: {rep_col}")
