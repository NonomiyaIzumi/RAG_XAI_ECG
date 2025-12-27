import time
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ========= Cấu hình =========
REPO_ROOT = Path(__file__).resolve().parents[2]
ARRAYS_PATH = str(Path(os.getenv("PREDICT_ARRAYS_PATH", str(REPO_ROOT / "22k_arrays.npy"))))
LABELS_PATH = str(Path(os.getenv("PREDICT_LABELS_PATH", str(REPO_ROOT / "22k_labels.npy"))))
CKPT_PATH   = str(Path(os.getenv("PREDICT_CKPT_PATH", str(REPO_ROOT / "Model" / "best_resnet50_fold4.pth"))))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

DISPLAY_NAMES = ["Norm", "MI", "STTC", "CD", "HYP"]

BATCH_SIZE = 64
NUM_CLASSES = 5
NUM_SAMPLES_TO_SHOW = 10

# ========= Dataset =========
class NpyECGImages(Dataset):
    def __init__(self, arrays, labels, transform):
        self.X = arrays
        self.y = labels.astype(np.float32)
        self.tf = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        x = self.tf(img)
        return x, idx

val_tf = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ========= Load dữ liệu =========
X = np.load(ARRAYS_PATH)
y = np.load(LABELS_PATH)
ds = NpyECGImages(X, y, val_tf)
loader = DataLoader(ds, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)

# ========= Model =========
def build_resnet50_head(num_classes=5):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def strip_module(state_dict):
    if next(iter(state_dict)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

model = build_resnet50_head(NUM_CLASSES)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = strip_module(ckpt.get("model_state_dict", ckpt))
model.load_state_dict(state, strict=False)
model.eval().to(DEVICE)

# ========= Inference =========
start = time.time()
all_probs = []
all_indices = []

with torch.no_grad():
    for xb, idxb in loader:
        xb = xb.to(DEVICE)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        all_probs.append(probs)
        all_indices.append(idxb.numpy())

all_probs = np.concatenate(all_probs)
all_indices = np.concatenate(all_indices)

elapsed = int((time.time() - start) * 1000)
print(f"1/1 [==============================] - 0s {elapsed}ms/step")

# ========= In kết quả =========
for i in range(min(NUM_SAMPLES_TO_SHOW, len(all_probs))):
    row = all_probs[i]
    probs_str = ", ".join(
        f"{n}: {p:.4f}" for n, p in zip(DISPLAY_NAMES, row)
    )
    print(f" Sample {i}: {probs_str}")

# ========= (Tuỳ chọn) Lưu toàn bộ prediction =========
out_path = Path(os.getenv("PREDICT_OUT_JSONL", str(REPO_ROOT / "all_predictions.jsonl")))
with out_path.open("w", encoding="utf-8") as f:
    for idx, row in zip(all_indices, all_probs):
        f.write(json.dumps({
            "index": int(idx),
            "probs": {n: float(p) for n, p in zip(DISPLAY_NAMES, row)}
        }) + "\n")

print(f"\n Saved predictions to {out_path}")
