# === BERTScore with library (robust, GPU) ===
import json, csv
import pandas as pd
from bert_score import score

GEMINI_PATH = "New_method_outputs.jsonl"
CSV_PATH    = "ptbxl_database.csv"
OUT_PATH    = "bertscore_new_method.csv"

# 1) Read JSONL + build map: index -> impression
imap = {}
with open(GEMINI_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        idx = rec.get("index", None)
        go  = rec.get("gemini_output", {})
        # lấy 'impression' trước; nếu thiếu thì nối các trường text khác
        if isinstance(go, dict):
            imp = go.get("impression")
            if not isinstance(imp, str):
                parts = []
                for k, v in go.items():
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                        parts.append("\n".join(v))
                imp = "\n".join(parts)
        else:
            imp = str(go)
        if idx is not None and isinstance(imp, str) and imp.strip():
            imap[int(idx)] = " ".join(imp.replace("\r"," ").replace("\n"," ").split())

# 2) Read CSV refs and align by ecg_id
df = pd.read_csv(CSV_PATH)
pairs = []
for _, row in df.iterrows():
    ecg_id = row.get("ecg_id")
    rep    = row.get("report")
    if pd.isna(ecg_id) or pd.isna(rep):
        continue
    ecg_id = int(ecg_id)
    if ecg_id in imap:
        ref = " ".join(str(rep).replace("\r"," ").replace("\n"," ").split())
        cand = imap[ecg_id]
        if ref and cand:
            pairs.append((ecg_id, cand, ref))

print("Số cặp:", len(pairs))
ids    = [p[0] for p in pairs]
cands  = [p[1] for p in pairs]
refs   = [p[2] for p in pairs]

# 3) Compute BERTScore
#    Chọn model_type ổn định, tránh dính AutoModel linh tinh.
#    Gợi ý: 'bert-base-multilingual-cased' (nhẹ, đa ngôn ngữ), hoặc 'xlm-roberta-large' nếu GPU khỏe.
MODEL_TYPE = "bert-base-multilingual-cased"  # đổi sang "xlm-roberta-large" nếu muốn
DEVICE     = "cuda"                           # "cuda" hoặc "cpu"
BATCH_SIZE = 32

P,R,F1 = score(
    cands, refs,
    model_type="xlm-roberta-large",
    num_layers=24,                 # XLM-R large có 24 layers -> last layer
    idf=False,
    rescale_with_baseline=False,
    device="cuda",
    batch_size=16,
    verbose=True
)


precision_mean = round(P.mean().item(),4)
recall_mean    = round(R.mean().item(),4)
f1_mean        = round(F1.mean().item(),4)

print("\n=== BERTScore (mean) ===")
print("Precision:", precision_mean)
print("Recall   :", recall_mean)
print("F1       :", f1_mean)

# 4) Save per-sample CSV
out_df = pd.DataFrame({
    "ecg_id": ids,
    "candidate_impression": cands,
    "reference_report": refs,
    "precision": [float(x) for x in P.tolist()],
    "recall":    [float(x) for x in R.tolist()],
    "f1":        [float(x) for x in F1.tolist()],
})
out_df.to_csv(OUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
print("Saved:", OUT_PATH)
