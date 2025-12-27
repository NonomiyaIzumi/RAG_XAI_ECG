import os, json, math, sys, types, gc
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
 
#=================== CONFIG ===================
ROOT_DIR    = Path(__file__).resolve().parent
INPUT_DIR   = str(Path(os.getenv("GRADCAM_INPUT_DIR", str(ROOT_DIR / "processed_images"))))
OUT_DIR     = str(Path(os.getenv("GRADCAM_OUT_DIR", str(ROOT_DIR / "gradcam_out"))))
ARCH        = "resnet50"
NUM_CLASSES = 5
WEIGHTS     = os.getenv("GRADCAM_WEIGHTS", str(ROOT_DIR / "Model" / "best_resnet50_fold4.pth"))
MULTI_LABEL = True         
CLASS_IDX   = None         
TOPK        = 1            
TARGET_LAYER= None          
RECURSIVE   = True          
INPUT_SIZE  = 224
ALPHA       = 0.45          
KEEP_RATIO  = True          
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# =================== Utils ===================
def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_image(img_path: Path, size: Tuple[int, int] = (224, 224), keep_ratio: bool = True):
    """
    Đọc ảnh -> tensor (1,3,H,W) normalize ImageNet + ảnh gốc RGB (H0,W0,3)
    """
    im = Image.open(str(img_path)).convert('RGB')
    orig = np.array(im)
    if keep_ratio:
        w, h = im.size
        side = max(w, h)
        new_im = Image.new('RGB', (side, side), (0, 0, 0))
        new_im.paste(im, ((side - w)//2, (side - h)//2))
        im = new_im
    im = im.resize(size, Image.BILINEAR)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = tfm(im).unsqueeze(0)
    return x, orig

# =================== Model ===================
def _load_any_state(m: nn.Module, weights_path: Optional[str]):
    if not weights_path:
        print(" Không có WEIGHTS, dùng random init.")
        return m
    p = Path(weights_path)
    if not p.exists():
        print(f" Không tìm thấy weights: {weights_path}")
        return m
    state = torch.load(str(p), map_location="cpu")
    # Hỗ trợ các format phổ biến
    candidates = []
    if isinstance(state, dict):
        for k in ["state_dict","model","model_state_dict","net","model_weights"]:
            if k in state and isinstance(state[k], dict):
                candidates.append(state[k])
    if not candidates and isinstance(state, dict):
        candidates.append(state)
    loaded = False
    for st in candidates:
        st = {k.replace("module.",""): v for k,v in st.items()}
        try:
            m.load_state_dict(st, strict=False)
            loaded = True
            break
        except Exception as e:
            # thử nốt
            pass
    print(" Load weights" if loaded else " Không khớp hoàn toàn, đã bỏ qua mismatch (strict=False).")
    return m

def build_model(arch: str, num_classes: int, weights: Optional[str] = None) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch in ["efficientnet_b0","efficientnet-b0"]:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Chưa hỗ trợ arch: {arch}")
    m = _load_any_state(m, weights)
    m.eval()
    return m

# =================== Grad-CAM core ===================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        self.model = model
        self.target_layer_name = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self.target_module = self._get_target_layer()
        self._register_hooks()

    def _get_target_layer(self) -> nn.Module:
        if self.target_layer_name:
            module = dict(self.model.named_modules()).get(self.target_layer_name)
            if module is None:
                raise ValueError(f"Không tìm thấy layer '{self.target_layer_name}'.")
            return module
        # Auto: Conv2d cuối cùng
        last_conv = None
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
                self.target_layer_name = n
        if last_conv is None:
            raise RuntimeError("Không tìm thấy Conv2d nào trong model.")
        print(f" Auto chọn target-layer: {self.target_layer_name}")
        return last_conv

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_module.register_forward_hook(fwd_hook))
        if hasattr(self.target_module, "register_full_backward_hook"):
            self.hook_handles.append(self.target_module.register_full_backward_hook(bwd_hook))
        else:
            self.hook_handles.append(self.target_module.register_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    @torch.no_grad()
    def _to_numpy_heatmap(self, cam: torch.Tensor, up_size: Tuple[int,int]) -> np.ndarray:
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-6)
        cam = cv2.resize(cam, up_size[::-1])  # cv2 expects (W,H)
        return cam

    def __call__(self, x: torch.Tensor, target_index: int, is_multilabel: bool,
                 input_size_hw: Tuple[int,int]) -> Tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x)  # (1,C)

        if is_multilabel:
            score = logits[0, target_index]
            prob = torch.sigmoid(score).item()
        else:
            score = logits[0, target_index]
            prob = F.softmax(logits, dim=1)[0, target_index].item()

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        grads = self.gradients       # (1,C,H,W)
        acts  = self.activations     # (1,C,H,W)
        weights = grads.mean(dim=(2,3), keepdim=True)    # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=False) # (1,H,W) -> squeeze
        cam = cam.squeeze(0)
        heatmap = self._to_numpy_heatmap(cam, input_size_hw)
        return heatmap, prob

# =================== Viz ===================
def overlay_heatmap_on_image(heatmap01: np.ndarray, orig_img: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h_uint8 = np.uint8(255 * heatmap01)
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)  # BGR
    H0, W0 = orig_img.shape[:2]
    h_color = cv2.resize(h_color, (W0, H0))
    base = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, h_color, alpha, 0)
    return overlay

# =================== Batch helpers ===================
def iter_image_paths(indir: Path, recursive: bool = False,
                     exts: Tuple[str,...]=('.png','.jpg','.jpeg','.bmp')) -> Iterable[Path]:
    if recursive:
        yield from (p for p in indir.rglob('*') if p.suffix.lower() in exts)
    else:
        yield from (p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in exts)

def decide_class_indices(scores: torch.Tensor, class_idx: Optional[int], topk: Optional[int]) -> List[int]:
    if class_idx is not None:
        return [class_idx]
    if topk is not None and topk > 0:
        vals, inds = torch.topk(scores, k=min(topk, scores.numel()))
        return inds.tolist()
    return [int(torch.argmax(scores).item())]

def process_one_image(img_path: Path, device, model: nn.Module, cam: GradCAM,
                      multi_label: bool, input_size: int, alpha: float,
                      keep_ratio: bool, out_dir: Path,
                      class_idx: Optional[int], topk: Optional[int]) -> Tuple[bool,str]:
    try:
        x, orig = load_image(img_path, size=(input_size, input_size), keep_ratio=keep_ratio)
        input_hw = (orig.shape[0], orig.shape[1])
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            scores = torch.sigmoid(logits)[0] if multi_label else F.softmax(logits, dim=1)[0]

        class_indices = decide_class_indices(scores, class_idx, topk)
        stem = img_path.stem
        ensure_dir(out_dir)

        for cls in class_indices:
            heatmap01, prob = cam(x, cls, multi_label, input_hw)
            overlay = overlay_heatmap_on_image(heatmap01, orig, alpha=alpha)
            out_overlay = out_dir / f"{stem}_cls{cls}_overlay.png"
            cv2.imwrite(str(out_overlay), overlay)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def run_folder(input_dir: str, out_dir: str,
               arch: str, num_classes: int, weights: Optional[str],
               multi_label: bool = False, class_idx: Optional[int] = None,
               topk: Optional[int] = 1, target_layer: Optional[str] = None,
               recursive: bool = True, input_size: int = 224, alpha: float = 0.45,
               keep_ratio: bool = True, device: Optional[str] = None):
    set_seed()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(arch, num_classes, weights).to(device)
    cam = GradCAM(model, target_layer=target_layer)

    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {in_dir}")
    paths = list(iter_image_paths(in_dir, recursive=recursive))
    if not paths:
        raise RuntimeError("Không tìm thấy ảnh (.png/.jpg/.jpeg/.bmp) trong thư mục.")

    print(f"🔎 Tìm thấy {len(paths)} ảnh. Bắt đầu tạo Grad-CAM...")
    ok, fail = 0, 0
    for p in paths:
        is_ok, msg = process_one_image(
            p, device, model, cam, multi_label, input_size, alpha, keep_ratio,
            Path(out_dir), class_idx, topk
        )
        if is_ok: ok += 1
        else:
            fail += 1
            print(f" {p}: {msg}")
    cam.remove_hooks()
    print(f"\n Hoàn tất. Thành công: {ok}, Lỗi: {fail}. Ảnh nằm tại: {out_dir}")
