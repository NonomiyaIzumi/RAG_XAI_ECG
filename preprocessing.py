import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import wfdb
from scipy.signal import iirnotch, filtfilt, butter
import scipy.signal as sg
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# ====== Đường dẫn ======
# Mặc định dùng PTB-XL folder nằm ngay trong repo. Có thể override bằng env var PTBXL_DIR.
ROOT_DIR = Path(__file__).resolve().parent
PTBXL_DIR = Path(os.getenv(
    "PTBXL_DIR",
    str(ROOT_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"),
)).expanduser()

# Metadata / labels
csv_path = Path(os.getenv("PTBXL_META_CSV", str(ROOT_DIR / "ptbxl_database_translated.csv")))
scp_path = Path(os.getenv("PTBXL_SCP_PATH", str(PTBXL_DIR / "scp_statements.csv")))
record_base = Path(os.getenv("PTBXL_RECORD_BASE", str(PTBXL_DIR)))

# Outputs (align with other scripts by default)
output_npy_dir = Path(os.getenv("OUT_NPY_DIR", str(ROOT_DIR / "ecg_processed")))
output_img_dir = Path(os.getenv("OUT_IMG_DIR", str(ROOT_DIR / "processed_images")))
output_lbl_dir = Path(os.getenv("OUT_LBL_DIR", str(ROOT_DIR / "processed_labels")))

output_npy_dir.mkdir(parents=True, exist_ok=True)
output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

# ====== Đọc metadata và nhãn ======
df = pd.read_csv(str(csv_path))
scp_statements = pd.read_csv(str(scp_path), index_col=0)
df = df[df['filename_hr'].notna()]
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
scp_statements_filtered = scp_statements[scp_statements.diagnostic == 1]

def aggregate_diagnostic_superclass(scp_code_dict):
    tmp = []
    for code in scp_code_dict:
        if code in scp_statements_filtered.index:
            superclass = scp_statements_filtered.loc[code].diagnostic_class
            tmp.append(superclass)
    return list(set(tmp))

df['superclass'] = df['scp_codes'].apply(aggregate_diagnostic_superclass)
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(df['superclass'])

# ====== Hàm xử lý tín hiệu ======
def smooth_signal(ecg, window_size=5):
    kernel = np.ones(window_size) / window_size
    return sg.convolve(ecg, kernel, mode='same')
# làm mượt bằng cách chia windowsize rồi lấy trung bình của windowsize đó, tức những rung động nhỏ như rung tay sẽ được chia trung bình=> làm mượt

def notch_filter(ecg, fs=100, freq=50.0, Q=30.0):
    b, a = sg.iirnotch(freq, Q, fs)
    return sg.filtfilt(b, a, ecg, axis=0)

def highpass_filter(ecg, fs=100, cutoff=0.5):
    b, a = sg.butter(3, cutoff / (0.5 * fs), btype='high')
    return sg.filtfilt(b, a, ecg, axis=0)

from scipy.signal import find_peaks

def extract_r_peaks(lead_signal, fs=100):
    r_peaks, _ = find_peaks(lead_signal, distance=fs*0.6, height=0.4)
    return r_peaks
#distance= khoang cach toi thieu giua 2 r peak, chi lay nhung dinh co bien do cao hon 0,4 mv

def extract_r_peaks_from_all_leads(record, fs=100):
    """
    record: np.array shape [12, 1000] – mỗi dòng là 1 lead
    returns: dict lead_idx -> list of R-peak indices
    """
    r_peak_dict = {}
    for i, lead in enumerate(record):
        peaks = extract_r_peaks(lead, fs)
        r_peak_dict[f"lead_{i+1}"] = peaks
    return r_peak_dict
#distance= khoang cach toi thieu giua 2 r peak, chi lay nhung dinh co bien do cao hon 0,4 mv

def preprocess_ecg_record(record, fs=100):
    processed = []
    r_peak_dict = {}

    for i, lead in enumerate(record.T):  # [1000, 12] → [12]
        x = smooth_signal(lead)
        x = notch_filter(x, fs)
        x = highpass_filter(x, fs)
        processed.append(x)

        r_peaks = extract_r_peaks(x, fs)
        r_peak_dict[f"lead_{i+1}"] = r_peaks

    return np.array(processed), r_peak_dict  # [12, 1000], {lead_i: [peaks]}

def save_ecg_image(ecg_array, save_path):
    fig, axes = plt.subplots(12, 1, figsize=(10, 10), sharex=True)
    for i in range(12):
        axes[i].plot(ecg_array[:, i], linewidth=0.8)
        axes[i].set_ylabel(f"Lead {i+1}")
        axes[i].set_yticks([])
        axes[i].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

df = df[df['ecg_id']]
success_ids = []
for idx, row in df.iterrows():
    ecg_id = row['ecg_id']
    file_path = str(record_base / Path(str(row['filename_lr'])).with_suffix(""))
    try:
        record, _ = wfdb.rdsamp(file_path)
        processed, r_peak_dict = preprocess_ecg_record(record, fs=100)  # shape [time, 12]

        np.save(str(output_npy_dir / f"{ecg_id}.npy"), processed.T)
        save_ecg_image(processed, str(output_img_dir / f"{ecg_id}.png"))
        np.save(str(output_lbl_dir / f"{ecg_id}.npy"), y_binary[idx])

        success_ids.append(ecg_id)
        print(f" {ecg_id} done")
    except Exception as e:
        print(f" Error with {ecg_id}: {e}")
