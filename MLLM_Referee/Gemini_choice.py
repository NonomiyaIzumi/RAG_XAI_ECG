import os
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable


# Đường dẫn file (sửa nếu khác)
BASELINE_PATH = ""
RIGHT_MIX_PATH = "New_method_outputs.jsonl"
PTBXL_PATH = "ptbxl_database.csv"
IMAGE_DIR = Path("processed_images")   

# Tên cột ID dùng để merge
ID_COL = "ecg_id"

# Cấu hình model Gemini
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 0.2
MAX_TOKENS = 1200

# Danh sách nhiều API key (xoay vòng khi quota)
API_KEYS = [

]

if not API_KEYS:
    raise ValueError("Vui lòng điền ít nhất 1 API key vào API_KEYS.")

_key_idx = 0  # index của key đang dùng

def _init_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
            "response_mime_type": "application/json",
        },
        safety_settings=None,
    )

model = _init_model(API_KEYS[_key_idx])

def _rotate_key() -> bool:
    """
    Chuyển sang API key tiếp theo trong API_KEYS.
    Trả về True nếu xoay được, False nếu đã hết key.
    """
    global _key_idx, model
    if _key_idx + 1 >= len(API_KEYS):
        return False
    _key_idx += 1
    model = _init_model(API_KEYS[_key_idx])
    print(f" Switched to API key #{_key_idx+1}/{len(API_KEYS)}")
    return True


def read_impression_jsonl(path,
                          json_id_field="index",
                          output_id_field=ID_COL):
    """
    Đọc file .jsonl theo cấu trúc:
    {
      "index": ...,
      ...
      "gemini_output": {
          "impression": "...",
          ...
      }
    }

    Trả về DataFrame với cột:
    - output_id_field (ecg_id)
    - impression (chuỗi tiếng Anh)
    """
    records = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            gem = obj.get("gemini_output", {})
            impression = gem.get("impression", "")

            rec = {
                output_id_field: obj[json_id_field],  # index -> ecg_id
                "impression": impression
            }
            records.append(rec)
    return pd.DataFrame(records)


# Đọc 2 file jsonl (baseline & right_cnn_mix)
baseline_df = read_impression_jsonl(BASELINE_PATH)
right_df    = read_impression_jsonl(RIGHT_MIX_PATH)

# Đổi tên cột impression để phân biệt
baseline_df = baseline_df.rename(columns={"impression": "impression_baseline"})
right_df    = right_df.rename(columns={"impression": "impression_right_cnn_mix"})

print("Baseline impressions:", baseline_df.shape)
print("Right_cnn_mix impressions:", right_df.shape)
print("\nBaseline head():")
print(baseline_df.head())

# Đọc PTB-XL CSV – để sep=None cho pandas tự đoán delimiter
ptbxl_df = pd.read_csv(PTBXL_PATH, sep=None, engine="python")

print("\nPTB-XL rows (full):", ptbxl_df.shape)
print("PTB-XL columns (full):", ptbxl_df.columns.tolist())

# Chỉ giữ lại 2 cột: ecg_id và report
ptbxl_df = ptbxl_df[[ID_COL, "report"]].copy()

print("\nPTB-XL sau khi lọc cột:", ptbxl_df.shape)
print("PTB-XL columns (filtered):", ptbxl_df.columns.tolist())

# Merge 2 file impression trên ecg_id
impressions_df = baseline_df.merge(
    right_df,
    on=ID_COL,
    how="inner"
)
print("\nMerged impressions:", impressions_df.shape)

# Merge impressions với PTB-XL (lúc này PTB-XL chỉ còn ecg_id + report)
full_df = ptbxl_df.merge(
    impressions_df,
    on=ID_COL,
    how="inner"
)

print("Full merged df:", full_df.shape)
full_df.head()


def safe_parse_json(text: str):
    """
    Cố gắng parse JSON dù Gemini có thể in thêm text.
    Nếu không parse được thì trả về {"raw_text": text}.
    """
    text = (text or "").strip()
    # Trường hợp chuỗi là JSON sạch
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Thử tìm đoạn {...} đầu tiên trong chuỗi
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # Bó tay -> lưu raw_text để debug sau
    return {"raw_text": text}


def _is_retryable_quota_error(data):
    """
    Kiểm tra xem có phải lỗi quota / rate limit không để xoay API key.
    """
    if not isinstance(data, dict):
        return False
    err = str(data.get("error", "")).upper()
    return any(k in err for k in ["DAILY_QUOTA_EXCEEDED", "SHOULD_ROTATE_KEY", "QUOTA", "RATE_LIMIT"])


def call_gemini_once(contents, retry=2, backoff=2.0):
    """
    Gọi Gemini 1 lần (với vài lần retry nhỏ khi lỗi server).
    Không xoay API key ở đây, chỉ trả về (ok, data, raw_text).

    - ok: True/False
    - data: dict từ safe_parse_json hoặc {"error": "..."}
    - raw_text: text thô từ Gemini để regex fallback.
    """
    last_err = None
    for i in range(retry):
        try:
            resp = model.generate_content(contents)
            raw_text = (getattr(resp, "text", "") or "").strip()
            data = safe_parse_json(raw_text)
            return True, data, raw_text
        except ResourceExhausted as e:
            s = str(e)
            if "quota" in s.lower():
                return False, {"error": "DAILY_QUOTA_EXCEEDED"}, None
            return False, {"error": "SHOULD_ROTATE_KEY", "detail": s}, None
        except (ServiceUnavailable, InternalServerError) as e:
            last_err = str(e)
            time.sleep(backoff * (i + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(backoff * (i + 1))
    return False, {"error": last_err}, None


def generate_with_rotation(contents, retry_per_key=2, backoff=2.0):
    """
    Gọi Gemini, nếu gặp quota/rate limit thì xoay API key và thử lại.
    Trả về resp (giống model.generate_content).
    """
    global model

    attempts = 0
    while True:
        ok, data, raw_text = call_gemini_once(contents, retry=retry_per_key, backoff=backoff)

        if ok:
            # giả lập object có .text để tương thích code cũ
            class _Resp:
                def __init__(self, text): self.text = text
            # raw_text có thể None hiếm gặp, nhưng cứ an toàn:
            return _Resp(raw_text or json.dumps(data, ensure_ascii=False))

        # not ok
        if _is_retryable_quota_error(data):
            rotated = _rotate_key()
            if not rotated:
                raise RuntimeError(f"Hết API key để xoay. Lỗi cuối: {data}")
            attempts += 1
            continue

        # lỗi khác (server/unknown) -> raise để bạn thấy rõ
        raise RuntimeError(f"Gemini call failed: {data}")


def judge_row_with_gemini(row):
    """
    Returns: "baseline" | "right_cnn_mix" | None
    """
    report_de = str(row["report"])
    cand1_en  = str(row["impression_baseline"])        # Candidate 1 (anonymous)
    cand2_en  = str(row["impression_right_cnn_mix"])   # Candidate 2 (anonymous)

    prompt = """
You are a cardiologist evaluating textual consistency.

You are given:
- A German ECG report.
- Two anonymous English candidate impressions: Candidate 1 and Candidate 2.
IMPORTANT: Do NOT prefer either candidate by default.

Task:
1) Translate both candidates into German (for fair comparison).
2) Compare each translated candidate against the German report.
3) You MUST choose exactly ONE winner: Candidate 1 or Candidate 2.
   - If both are weak, choose the more specific one.
   - Never output “tie”, “neither”, or any third option.

Output rules:
- Output EXACTLY one JSON object, no extra text:
  {"chosen_file":"baseline"} or {"chosen_file":"right_cnn_mix"}

Mapping:
- If Candidate 1 wins -> "baseline"
- If Candidate 2 wins -> "right_cnn_mix"
"""

    contents = [
        prompt,
        "\n=== German ECG report ===\n", report_de,
        "\n=== Candidate 1 (English) ===\n", cand1_en,
        "\n=== Candidate 2 (English) ===\n", cand2_en,
    ]

    try:
        resp = generate_with_rotation(contents, retry_per_key=2, backoff=2.0)
        raw_text = (getattr(resp, "text", "") or "").strip()

        data = safe_parse_json(raw_text)
        if not data:
            print("Invalid response (no parsable JSON):", raw_text[:250])
            return None

        choice = data.get("chosen_file")
        if choice not in ("baseline", "right_cnn_mix"):
            print("Invalid choice:", raw_text[:250])
            return None

        return choice

    except Exception as e:
        print("Error calling/parsing Gemini:", e)
        return None



MAX_SAMPLES = None  

n_rows = len(full_df)
if MAX_SAMPLES is not None:
    n_rows = min(n_rows, MAX_SAMPLES)

print(f"Will evaluate {n_rows} records.")

scores = {"baseline": 0, "right_cnn_mix": 0, "none": 0}
choices = []

for idx in range(n_rows):
    row = full_df.iloc[idx]
    choice = judge_row_with_gemini(row)
    choices.append(choice)

    if choice is None:
        scores["none"] += 1
    else:
        scores[choice] += 1

    if (idx + 1) % 10 == 0 or idx == n_rows - 1:
        print(f"Processed {idx + 1}/{n_rows} records...")

print("\n===== FINAL SCORES =====")
print(f"Total evaluated      : {n_rows}")
print(f"baseline chosen      : {scores['baseline']}")
print(f"right_cnn_mix chosen : {scores['right_cnn_mix']}")
print(f"no decision / errors : {scores['none']}")

result_df = full_df.iloc[:n_rows].copy()
result_df["gemini_choice"] = choices

OUTPUT_CSV = "gemini_impression_comparison_results.csv"
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved detailed results to: {OUTPUT_CSV}")
