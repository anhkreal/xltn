import os
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

FS = 16000
VOWELS = ['a', 'e', 'i', 'o', 'u']
RESULT_DIR = 'results'
TEST_SAMPLE_DIR = os.path.join('signals', 'NguyenAmKiemThu-16k', '25MLM')

# Chọn 1 file wav bất kỳ
sample_wav = None
for f in os.listdir(TEST_SAMPLE_DIR):
    if f.endswith('.wav'):
        sample_wav = os.path.join(TEST_SAMPLE_DIR, f)
        break
if sample_wav is None:
    raise FileNotFoundError('Không tìm thấy file wav trong 25MLM')

print(f"Sử dụng file test: {sample_wav}")

y, sr = librosa.load(sample_wav, sr=FS)

# --- FFT MODEL ---
try:
    with open(os.path.join(RESULT_DIR, 'Task1_Best_FFT_Model.pkl'), 'rb') as f:
        fft_model = pickle.load(f)
    n_fft = fft_model['n_fft']
    models = fft_model['models']
    # Tính vector đặc trưng
    def compute_fft_vector(y, n_fft):
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)))
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=int(0.01*FS)))
        vec = np.mean(S, axis=1)
        return vec / (np.max(vec) + 1e-8)
    # Cắt 1/3 giữa
    length = len(y)
    start = int(length / 3)
    end = int(2 * length / 3)
    y_steady = y[start:end] if end - start >= 512 else y
    feat = compute_fft_vector(y_steady, n_fft)
    dists = {v: np.linalg.norm(feat - models[v]) for v in VOWELS}
    pred_fft = min(dists, key=dists.get)
    print(f"[FFT] Dự đoán: /{pred_fft}/ (Khoảng cách: {dists[pred_fft]:.4f})")
except Exception as e:
    print(f"[FFT] Không load được model: {e}")

# --- KMEANS MODEL ---
try:
    with open(os.path.join(RESULT_DIR, 'Task2_Best_KMeans_Model.pkl'), 'rb') as f:
        kmeans_model = pickle.load(f)
    codebooks = kmeans_model['codebooks']
    k = kmeans_model['k']
    # MFCC
    mfcc = librosa.feature.mfcc(y=y_steady, sr=FS, n_mfcc=13, n_fft=1024, hop_length=int(0.01*FS)).T
    min_dist = float('inf'); best_v = None
    for v in VOWELS:
        centers = codebooks[v]
        dists = np.min(np.linalg.norm(mfcc[:, None] - centers[None, :], axis=2), axis=1)
        avg_dist = np.mean(dists)
        if avg_dist < min_dist:
            min_dist = avg_dist
            best_v = v
    print(f"[K-means] Dự đoán: /{best_v}/ (Khoảng cách TB: {min_dist:.4f})")
except Exception as e:
    print(f"[K-means] Không load được model: {e}")

# --- HMM MODEL ---
try:
    with open(os.path.join(RESULT_DIR, 'Task3_Best_HMM_Model.pkl'), 'rb') as f:
        hmm_model = pickle.load(f)
    models = hmm_model['models']
    mfcc = librosa.feature.mfcc(y=y, sr=FS, n_mfcc=hmm_model['mfcc_dim'], n_fft=1024, hop_length=int(0.01*FS)).T
    best_score = -float('inf'); best_v = None
    for v in VOWELS:
        try:
            score = models[v].score(mfcc)
            if score > best_score:
                best_score = score
                best_v = v
        except Exception:
            continue
    print(f"[HMM] Dự đoán: /{best_v}/ (Score: {best_score:.2f})")
except Exception as e:
    print(f"[HMM] Không load được model: {e}")
