import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

FS = 16000
VOWELS = ['a', 'e', 'i', 'o', 'u']
TRAIN_DIR = os.path.join('signals', 'NguyenAmHuanLuyen-16k')
TEST_DIR = os.path.join('signals', 'NguyenAmKiemThu-16k')
RESULT_DIR = 'results'

os.makedirs(RESULT_DIR, exist_ok=True)

# Utils

def extract_stable_segment(y, sr, strict_mode=True):
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) == 0:
        return y
    if not strict_mode:
        return y_trimmed
    length = len(y_trimmed)
    start = int(length / 3)
    end = int(2 * length / 3)
    if end - start < 512:
        return y_trimmed
    return y_trimmed[start:end]

def load_data_paths(root_dir):
    data_paths = {v: [] for v in VOWELS}
    if not os.path.exists(root_dir):
        return data_paths
    for person in os.listdir(root_dir):
        path = os.path.join(root_dir, person)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.wav'):
                    v = file.split('.')[0].lower()
                    if v in VOWELS:
                        data_paths[v].append(os.path.join(path, file))
    return data_paths

# Features

def compute_fft_vector(file_path, n_fft):
    y, _ = librosa.load(file_path, sr=FS)
    y_steady = extract_stable_segment(y, FS, strict_mode=True)
    if len(y_steady) < n_fft:
        y_steady = np.pad(y_steady, (0, n_fft - len(y_steady)))
    S = np.abs(librosa.stft(y_steady, n_fft=n_fft, hop_length=int(0.01*FS)))
    vec = np.mean(S, axis=1)
    return vec / (np.max(vec) + 1e-8)

# Plot helpers

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=VOWELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VOWELS, yticklabels=VOWELS)
    plt.title(title); plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.savefig(os.path.join(RESULT_DIR, filename)); plt.close()

def save_feature_plot(vectors_dict, title, xlabel, ylabel, filename, x_axis_values=None):
    plt.figure(figsize=(10, 6))
    first_key = list(vectors_dict.keys())[0]
    dim = len(vectors_dict[first_key])
    x_axis = np.arange(dim) if x_axis_values is None else x_axis_values
    for v in VOWELS:
        if v in vectors_dict:
            plt.plot(x_axis, vectors_dict[v], label=f'/{v}/', linewidth=1.5, alpha=0.8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename)); plt.close()

# Task 1 runner

def run_task_1_fft(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 1: FFT (N_FFT: 512, 1024, 2048)")
    results = []
    n_fft_list = [512, 1024, 2048]
    best_acc, best_y_true, best_y_pred, best_n = 0, [], [], 0
    best_models = None

    for n_fft in n_fft_list:
        models = {}
        for v in VOWELS:
            vecs = [compute_fft_vector(p, n_fft) for p in train_paths[v]]
            models[v] = np.mean(vecs, axis=0)
        freq_axis = np.linspace(0, FS/2, len(next(iter(models.values()))))
        save_feature_plot(models, f"Phổ FFT Trung Bình (N={n_fft})", "Tần số (Hz)", "Biên độ", f"Task1_Vectors_N{n_fft}.png", x_axis_values=freq_axis)

        y_true, y_pred = [], []
        for v_true in VOWELS:
            for p in test_paths[v_true]:
                feat = compute_fft_vector(p, n_fft)
                best_v = min(models, key=lambda k: np.linalg.norm(feat - models[k]))
                y_true.append(v_true); y_pred.append(best_v)

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"   N_FFT={n_fft}: Acc={acc:.2f}%")
        results.append({'N_FFT': n_fft, 'Accuracy': acc})
        if acc > best_acc:
            best_acc = acc; best_n = n_fft; best_y_true, best_y_pred = y_true, y_pred; best_models = models

    print("\nBẢNG KẾT QUẢ ACCURACY (FFT)")
    print("+---------+----------+")
    print("| N_FFT   | Accuracy |")
    print("+---------+----------+")
    for r in results:
        print(f"| {r['N_FFT']:<7} | {r['Accuracy']:<8.2f} |")
    print("+---------+----------+\n")

    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task1_Results.csv'), index=False)
    save_confusion_matrix(best_y_true, best_y_pred, f"CM Bài 1 (Best N={best_n})", "Task1_Best_CM.png")

    # Accuracy bar
    df1 = pd.DataFrame(results)
    plt.figure(figsize=(7,5)); sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df1, x='N_FFT', y='Accuracy', palette='Blues', edgecolor='black')
    plt.title("Accuracy theo N_FFT (FFT)", fontsize=15, fontweight='bold'); plt.xlabel("N_FFT"); plt.ylabel("Accuracy (%)"); plt.ylim(0,100)
    for p in ax.patches:
        h = p.get_height();
        if not np.isnan(h): ax.annotate(f"{h:.2f}", (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0,3), textcoords='offset points')
    plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR, 'Task1_Accuracy_Bar.png'), dpi=120); plt.close()

    # Save best model
    with open(os.path.join(RESULT_DIR, 'Task1_Best_FFT_Model.pkl'), 'wb') as f:
        pickle.dump({'models': best_models, 'n_fft': best_n, 'accuracy': best_acc}, f)
    print(f"-> Đã lưu model FFT tốt nhất (N_FFT={best_n}, Acc={best_acc:.2f}%)")

if __name__ == '__main__':
    train_paths = load_data_paths(TRAIN_DIR)
    test_paths = load_data_paths(TEST_DIR)
    if any(len(x) > 0 for x in train_paths.values()):
        run_task_1_fft(train_paths, test_paths)
        print(f"\n[DONE] Kết quả đã lưu tại thư mục '{RESULT_DIR}'.")
    else:
        print("Lỗi: Không tìm thấy dữ liệu wav.")
