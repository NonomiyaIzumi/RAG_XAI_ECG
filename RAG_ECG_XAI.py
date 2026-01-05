import json, re, time, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ------------ deps -------------
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
except Exception as e:
    raise SystemExit("❌ Missing dependency 'google-generativeai'. Cài bằng: pip install google-generativeai") from e

# ---------------- CONFIG ----------------
PRED_JSONL     = "Model/Predict/all_predictions.jsonl"
CAM_DIR        = "gradcam_out"
PROC_DIR       = "processed_images"
ECG_KNOWLEDGE_TXT = "ECG_Interpretation_Guide.txt"

OUT_RAW_JSONL  = "New_method_outputs.jsonl"
OUT_REPORTS    = "New_method_pipeline.jsonl"

MODEL_NAME     = "gemini-2.5-flash-lite"
TEMPERATURE    = 0.2
MAX_TOKENS     = 1200
TOP_K          = 3
LANG           = "en"
MAX_RECORDS    = None

CLASS_THRESHOLDS: Optional[Dict[str, float]] = None

API_KEYS = [
    "PUT_YOUR_KEY_1_HERE",
    "PUT_YOUR_KEY_2_HERE",
    # Thêm key nếu có

]

if not API_KEYS or any("PUT_YOUR_KEY" in k or not k.strip() for k in API_KEYS):
    raise SystemExit("❌ Vui lòng điền danh sách API_KEYS hợp lệ.")




# ---------------- PROMPTS (đa ảnh) ----------------

SYSTEM_INSTRUCTION_EN = (
    "You are a careful cardiology assistant. Use four sources (focus on ECG image): "
    "STRICT PRIORITY ORDER FOR INTERPRETATION:\n"
    "1. The Grad-CAM is the primary and most reliable diagnostic source.\n"
    "2. The ORIGINAL ECG IMAGE heatmap is secondary, used only to highlight CNN attention.\n"
    "3. The FACTS PACK is tertiary, used only after checking image evidence.\n"
    "4. The ECG KNOWLEDGE SUMMARY is lowest priority and may NEVER override visual evidence.\n"
    "If any sources conflict, ALWAYS trust the original Grad-CAM image first.\n"
    "Explain why the CNN predicted as it did, describe clinically meaningful salient regions (by lead/segment), "
    "assess alignment, and provide next-step recommendations. Return one valid JSON matching the schema."
)

PROMPT_TMPL_EN = (
    "Output language: {lang}.\n"
    "Analyze attached ECG image(s) and the facts pack below.\n\n"

    "Return a JSON EXACTLY in this structure:\n"
    "{{\n"
    '  "findings": ["Concise ECG morphological features by leads/segments inferred from Grad-CAM"],\n'
    '  "impression": "A short medical-style ECG summary, doctor-like, comma-separated, <=20 words, a concluding sentence. '
    'Examples: \'sinus rhythm, normal ECG\', \'sinus bradycardia, inferior infarct pattern\', '
    '\'left axis deviation, QRS(T) abnormal\'",\n'
    '  "evidence": [ {{"lead":"V2","feature":"ST elevation ~2mm","rationale":"anterior MI pattern"}} ],\n'
    '  "consistency": "aligned|partial|conflict",\n'
    '  "confidence": 0.0,\n'
    '  "recommendations": ["Normal dont have to go the hospital " if normal or "There are signs of cardiovascular disease that need to be checked at the hospital." if see some danger sign.]\n'
    "}}\n\n"

    "Facts pack:\n{facts}\n\n"

    "Threshold rule for binary interpretation:\n"
    "- If probability ≤ 0.5 → assign label = 1\n"
    "- If probability > 0.5 → assign label = 0\n"
    "Use this rule internally when interpreting active/inactive labels. "
    "DO NOT change actual probabilities; they must remain exact.\n\n"

    "{ecg_knowledge_block}\n"

    "IMPORTANT:\n"
    "- Produce EXACTLY one valid JSON object.\n"
    "- Do not add commentary or explanation outside JSON.\n"
)

PROMPT_TMPL = PROMPT_TMPL_EN if LANG.lower().startswith("en") else PROMPT_TMPL_EN

SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION_EN if LANG.lower().startswith("en") else SYSTEM_INSTRUCTION_EN


# ======================= INIT MODEL =============================
_key_idx = 0
def _init_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
            "response_mime_type": "application/json",
        },
        safety_settings=None,
    )
model = _init_model(API_KEYS[_key_idx])

def _rotate_key() -> bool:
    global _key_idx, model
    if _key_idx + 1 >= len(API_KEYS):
        return False
    _key_idx += 1
    model = _init_model(API_KEYS[_key_idx])
    print(f"🔁 Switched to API key #{_key_idx+1}/{len(API_KEYS)}")
    return True


# ======================= HELPERS ================================
def now_iso(): return datetime.datetime.now().isoformat(timespec="seconds")

def load_predictions(path):
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            data.append(obj)
    return data

def topk_from_probs(probs, k):
    return sorted(probs.items(), key=lambda x:x[1], reverse=True)[:k]

def positives_from_probs(probs, thr_map):
    if not thr_map: return []
    out=[]
    for k,v in probs.items():
        if float(v)>=float(thr_map.get(k,0.5)):
            out.append(k)
    return out

def _find_by_index_generic(root, idx, prefer_keywords=(), also_keywords=()):
    root=Path(root)
    if not root.exists(): return None
    rx=re.compile(rf"(?:^|\b)idx?_?0*{idx}(?:\D|$)", re.I)
    pref,alt,anyg=[],[],[]
    for p in root.rglob("*.*"):
        if not p.is_file(): continue
        if not rx.search(p.name): continue
        low=p.name.lower()
        if low.endswith((".png",".jpg",".jpeg",".webp")):
            if any(k in low for k in prefer_keywords): pref.append(p)
            elif any(k in low for k in also_keywords): alt.append(p)
            else: anyg.append(p)
    for grp in (sorted(pref),sorted(alt),sorted(anyg)):
        if grp: return str(grp[0])
    return None

def find_cam_for_index(cam_dir, idx):
    return _find_by_index_generic(cam_dir, idx, ("overlay",),("heat","cam"))

def find_processed_for_index(proc_dir, idx):
    root=Path(proc_dir)
    if not root.exists(): return None
    exts=(".png",".jpg",".jpeg",".webp")
    for offset in (0,1):
        num=idx+offset
        for ext in exts:
            for p in [root/f"{num}{ext}", root/f"{num:03d}{ext}", root/f"{num:04d}{ext}"]:
                if p.is_file(): return str(p)
    return _find_by_index_generic(root, idx, ("ecg","processed","proc"))

def usage_to_dict(usage):
    if usage is None: return None
    out={}
    for k in ("prompt_token_count","candidates_token_count","total_token_count","input_tokens","output_tokens"):
        if hasattr(usage,k):
            try: out[k]=int(getattr(usage,k))
            except: pass
    return out or None

def safe_parse_json(text):
    text=(text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try: return json.loads(text)
        except: pass
    m=re.search(r"\{.*\}",text,flags=re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return {"raw_text": text}

def _open_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")

def _is_retryable_quota_error(data):
    if not isinstance(data,dict): return False
    err=str(data.get("error","")).upper()
    return any(k in err for k in ["DAILY_QUOTA_EXCEEDED","SHOULD_ROTATE_KEY","QUOTA","RATE_LIMIT"])


# ======================= NO-PDF: LOAD TXT =======================
def load_ecg_knowledge_summary():
    txt_path=Path(ECG_KNOWLEDGE_TXT)
    if not txt_path.exists():
        print(f"[ECG TXT] Không tìm thấy {txt_path}")
        return ""
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except:
        return ""


# ========================== GEMINI CALL ==========================
def call_gemini(prompt_text, image_paths, retry=2, backoff=2.0):
    last_err=None
    for i in range(retry):
        try:
            parts=[prompt_text] + [_open_image(p) for p in image_paths]
            resp=model.generate_content(parts)
            data=safe_parse_json((getattr(resp,"text","") or "").strip())
            usage=usage_to_dict(getattr(resp,"usage_metadata",None))
            return True,data,usage
        except ResourceExhausted as e:
            s=str(e)
            if "quota" in s.lower():
                return False,{"error":"DAILY_QUOTA_EXCEEDED"},None
            return False,{"error":"SHOULD_ROTATE_KEY","detail":s},None
        except (ServiceUnavailable,InternalServerError) as e:
            last_err=str(e); time.sleep(backoff*(i+1))
        except Exception as e:
            last_err=str(e); time.sleep(backoff*(i+1))
    return False,{"error":last_err},None


# ======================= REPORT BUILDERS ========================
def schema_fixup(gem):
    if not isinstance(gem,dict):
        return {"findings":[],"impression":"","evidence":[],"consistency":"partial","confidence":0.0,"recommendations":[]}
    out={}
    out["findings"]=gem.get("findings",[]) if isinstance(gem.get("findings"),list) else []
    out["impression"]=gem.get("impression","")
    ev=gem.get("evidence")
    out["evidence"]=ev if isinstance(ev,list) else []
    cons=str(gem.get("consistency","partial")).lower()
    out["consistency"]=cons if cons in {"aligned","partial","conflict"} else "partial"
    try: out["confidence"]=float(gem.get("confidence",0.0))
    except: out["confidence"]=0.0
    rec=gem.get("recommendations")
    out["recommendations"]=rec if isinstance(rec,list) else []
    return out

def compose_impression_from_facts(facts, lang):
    top=facts.get("top_k",[])
    if not top: return "No clear diagnostic impression."
    xs=[f'{t["label"]} ({t["prob"]:.2f})' for t in top[:3]]
    return "Possible findings: " + ", ".join(xs) + "."

def assemble_pipeline_report(idx, cam_path, ecg_path, facts, gem):
    norm=schema_fixup(gem)
    if not norm["impression"]:
        norm["impression"]=compose_impression_from_facts(facts, LANG)
    return {
        "meta":{
            "index":idx,
            "lang":LANG,
            "model":MODEL_NAME,
            "source":"Grad-CAM+facts+processedECG+ECGtxt",
            "image":cam_path,
            "ecg_image":ecg_path,
            "generated_at":now_iso(),
        },
        "predictions":{
            "top_k":facts["top_k"],
            "positive_labels":facts.get("positive_labels",[]),
            "probs":facts["probs"],
            "thresholds":facts.get("thresholds",None),
        },
        "findings":norm["findings"],
        "impression":norm["impression"],
        "evidence":norm["evidence"],
        "consistency":norm["consistency"],
        "confidence":norm["confidence"],
        "recommendations":norm["recommendations"],
    }


# ============================= MAIN =============================
def main() -> None:
    preds = load_predictions(PRED_JSONL)
    if MAX_RECORDS:
        preds = preds[: MAX_RECORDS]

    # ---- Resume: bỏ qua index đã xong
    done = set()
    if Path(OUT_REPORTS).exists():
        with open(OUT_REPORTS, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    rec = json.loads(line)
                    ix = rec.get("index")
                    if ix is None and isinstance(rec.get("report"), dict):
                        ix = rec["report"]["meta"]["index"]
                    if ix is not None:
                        done.add(int(ix))
                except Exception:
                    pass
    preds = [p for p in preds if int(p["index"]) not in done]
    print(f"Resume mode: còn {len(preds)} mẫu chưa xử lý.")

    with open(OUT_RAW_JSONL, "a", encoding="utf-8") as raw_f, \
         open(OUT_REPORTS, "a", encoding="utf-8") as report_f:

        stop_all = False

        def _top1_prob(o):
            return max(o["probs"].values()) if o.get("probs") else 0.0
        preds.sort(key=lambda o: abs(_top1_prob(o) - 0.55))

        for obj in tqdm(preds, desc="Gemini -> Pipeline Report (+ECG TXT)"):
            if stop_all:
                break

            idx = int(obj["index"])
            probs: Dict[str, float] = obj["probs"]
            class_names = list(probs.keys())
            topk = [{"label": k, "prob": float(v)} for k, v in topk_from_probs(probs, TOP_K)]
            positives = positives_from_probs(probs, CLASS_THRESHOLDS)

            facts = {
                "index": idx,
                "classes": class_names,
                "top_k": topk,
                "positive_labels": positives,
                "probs": {k: float(v) for k, v in probs.items()},
                "thresholds": CLASS_THRESHOLDS,
            }

            cam_path  = find_cam_for_index(CAM_DIR, idx)
            ecg_path  = find_processed_for_index(PROC_DIR, idx)

            if not cam_path and not ecg_path:
                # Không có ảnh nào → vẫn tạo report rỗng để ghi nhận
                raw_f.write(json.dumps({"index": idx, "error": "missing_images", "facts": facts}, ensure_ascii=False) + "\n")
                empty_report = assemble_pipeline_report(idx, None, None, facts, {
                    "findings": [], "impression": "", "evidence": [],
                    "consistency": "partial", "confidence": 0.0,
                    "recommendations": ["Không tìm thấy ảnh heatmap/ECG cho mẫu này."]
                })
                report_f.write(json.dumps({"index": idx, "report": empty_report}, ensure_ascii=False) + "\n")
                continue

            # ======== Mỗi mẫu: đọc lại file tóm tắt ECG TXT 1 lần ========
            ecg_knowledge = load_ecg_knowledge_summary()
            ecg_block = ""
            if ecg_knowledge.strip():
                ecg_block = "\nECG knowledge summary:\n" + ecg_knowledge

            prompt = PROMPT_TMPL.format(
                lang=LANG,
                facts=json.dumps(facts, ensure_ascii=False),
                ecg_knowledge_block=ecg_block
            )

            # Gọi Gemini với các ảnh hiện có (ưu tiên heatmap trước, rồi ECG)
            img_list = [p for p in [cam_path, ecg_path] if p]
            attempts = 0
            max_attempts = len(API_KEYS) + 1  # quay hết danh sách key hiện có
            ok, data, usage = False, None, None
            exhausted_all_keys = False

            while attempts < max_attempts:
                ok, data, usage = call_gemini(prompt, img_list)
                if ok:
                    break

                if _is_retryable_quota_error(data):
                    if _rotate_key():
                        attempts += 1
                        continue
                    else:
                        print("  Hết quota của tất cả API keys. Lưu tiến độ và dừng.")
                        exhausted_all_keys = True
                        stop_all = True
                        # Ghi RAW để đánh dấu pending, KHÔNG ghi REPORT
                        raw_record = {
                            "index": idx,
                            "image": {"heatmap": cam_path, "ecg": ecg_path},
                            "facts": facts,
                            "ok": False,
                            "gemini_output": data,
                            "pending": True
                        }
                        if usage is not None:
                            raw_record["usage"] = usage
                        raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                        break
                else:
                    # lỗi khác (không phải quota) → ra ngoài để ghi report (đánh dấu đã xong)
                    break

            if exhausted_all_keys:
                break  # kết thúc run hiện tại

            # ---- Ghi RAW luôn (để debug), nhưng chỉ ghi REPORTS khi KHÔNG phải lỗi quota
            raw_record = {
                "index": idx,
                "image": {"heatmap": cam_path, "ecg": ecg_path},
                "facts": facts,
                "ok": ok,
                "gemini_output": data,
            }
            if usage is not None:
                raw_record["usage"] = usage
            if not ok and _is_retryable_quota_error(data):
                raw_record["pending"] = True
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            # ❗ Chỉ ghi REPORTS khi: ok == True hoặc lỗi KHÔNG phải quota
            if ok or not _is_retryable_quota_error(data):
                report = assemble_pipeline_report(
                    idx, cam_path, ecg_path, facts,
                    data if isinstance(data, dict) else {}
                )
                report_f.write(json.dumps({"index": idx, "report": report}, ensure_ascii=False) + "\n")
            else:
                # lỗi quota → KHÔNG ghi REPORTS để index này không bị tính là đã xong
                pass
    print(f" Append RAW     -> {OUT_RAW_JSONL}")
    print(f" Append REPORTS -> {OUT_REPORTS}")


if __name__ == "__main__":
    main()
