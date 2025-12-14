import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
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

def compute_mfcc_vectors(file_path, dim_type=13, n_fft=1024, strict=True):
    y, _ = librosa.load(file_path, sr=FS)
    y_steady = extract_stable_segment(y, FS, strict_mode=strict)
    mfcc = librosa.feature.mfcc(y=y_steady, sr=FS, n_mfcc=13, n_fft=n_fft, hop_length=int(0.01*FS))
    if dim_type == 13:
        return mfcc.T
    delta1 = librosa.feature.delta(mfcc)
    if dim_type == 26:
        return np.vstack([mfcc, delta1]).T
    delta2 = librosa.feature.delta(mfcc, order=2)
    if dim_type == 39:
        return np.vstack([mfcc, delta1, delta2]).T
    return mfcc.T

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

# Task 2 runner

def run_task_2_kmeans(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 2: K-MEANS (K: 2, 3, 4, 5)")
    k_list = [2, 3, 4, 5]

    print("   [Cache] Extracting MFCC-13 (Strict Mode)...")
    train_feats = {v: [compute_mfcc_vectors(p, 13, strict=True) for p in train_paths[v]] for v in VOWELS}
    test_feats = {v: [compute_mfcc_vectors(p, 13, strict=True) for p in test_paths[v]] for v in VOWELS}

    global_means = {v: np.mean(np.vstack(train_feats[v]), axis=0) for v in VOWELS}
    save_feature_plot(global_means, "Vector MFCC Trung Bình", "MFCC Index", "Value", "Task2_Global_MFCC.png")

    results = []
    best_acc, best_y_true, best_y_pred, best_k = 0, [], [], 0
    best_codebooks = None

    for k in k_list:
        codebooks = {}
        for v in VOWELS:
            X = np.vstack(train_feats[v])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            codebooks[v] = kmeans.cluster_centers_

        y_true, y_pred = [], []
        for v_true in VOWELS:
            for frames in test_feats[v_true]:
                min_dist = float('inf'); best_v = None
                for v_model, centers in codebooks.items():
                    dists = np.min(np.linalg.norm(frames[:, None] - centers[None, :], axis=2), axis=1)
                    avg_dist = np.mean(dists)
                    if avg_dist < min_dist: min_dist = avg_dist; best_v = v_model
                y_true.append(v_true); y_pred.append(best_v)

        acc = accuracy_score(y_true, y_pred) * 100
        print(f"   K={k}: Acc={acc:.2f}%")
        results.append({'K': k, 'Accuracy': acc})
        if acc > best_acc:
            best_acc = acc; best_k = k; best_y_true, best_y_pred = y_true, y_pred; best_codebooks = codebooks

    print("\nBẢNG KẾT QUẢ ACCURACY (K-means)")
    print("+-----+----------+")
    print("| K   | Accuracy |")
    print("+-----+----------+")
    for r in results:
        print(f"| {r['K']:<3} | {r['Accuracy']:<8.2f} |")
    print("+-----+----------+\n")

    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task2_Results.csv'), index=False)
    save_confusion_matrix(best_y_true, best_y_pred, f"CM Bài 2 (Best K={best_k})", "Task2_Best_CM.png")

    # Accuracy bar
    df2 = pd.DataFrame(results)
    plt.figure(figsize=(7,5)); sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df2, x='K', y='Accuracy', palette='Oranges', edgecolor='black')
    plt.title("Accuracy theo K (K-means)", fontsize=15, fontweight='bold'); plt.xlabel("K"); plt.ylabel("Accuracy (%)"); plt.ylim(0,100)
    for p in ax.patches:
        h = p.get_height();
        if not np.isnan(h): ax.annotate(f"{h:.2f}", (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0,3), textcoords='offset points')
    plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR, 'Task2_Accuracy_Bar.png'), dpi=120); plt.close()

    # Save best model
    with open(os.path.join(RESULT_DIR, 'Task2_Best_KMeans_Model.pkl'), 'wb') as f:
        pickle.dump({'codebooks': best_codebooks, 'k': best_k, 'accuracy': best_acc}, f)
    print(f"-> Đã lưu model K-means tốt nhất (K={best_k}, Acc={best_acc:.2f}%)")

if __name__ == '__main__':
    train_paths = load_data_paths(TRAIN_DIR)
    test_paths = load_data_paths(TEST_DIR)
    if any(len(x) > 0 for x in train_paths.values()):
        run_task_2_kmeans(train_paths, test_paths)
        print(f"\n[DONE] Kết quả đã lưu tại thư mục '{RESULT_DIR}'.")
    else:
        print("Lỗi: Không tìm thấy dữ liệu wav.")
