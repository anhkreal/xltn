import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from hmmlearn import hmm
import pandas as pd
import warnings
import pickle

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# --- CẤU HÌNH HỆ THỐNG ---
FS = 16000
VOWELS = ['a', 'e', 'i', 'o', 'u']
TRAIN_DIR = os.path.join('signals', 'NguyenAmHuanLuyen-16k')
TEST_DIR = os.path.join('signals', 'NguyenAmKiemThu-16k')
RESULT_DIR = 'results' 

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# --- 1. XỬ LÝ TÍN HIỆU & PHÂN ĐOẠN ---

def extract_stable_segment(y, sr, strict_mode=True):
    """
    Trích xuất vùng tín hiệu.
    - strict_mode=True: Lấy đúng 1/3 đoạn giữa (Yêu cầu Bài 1, 2)
    - strict_mode=False: Chỉ cắt khoảng lặng đầu/cuối (Tối ưu cho Bài 3 HMM)
    """
    # Bước 1: Cắt bỏ khoảng lặng (Trim silence)
    # top_db=30: Ngưỡng cắt chặt hơn để loại bỏ nhiễu nền
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    if len(y_trimmed) == 0: return y
    
    if not strict_mode:
        # Bài 3: Lấy toàn bộ phần hữu thanh để HMM học biến thiên
        return y_trimmed
    
    # Bài 1, 2: Lấy 1/3 ở giữa để có đặc trưng tĩnh ổn định nhất
    length = len(y_trimmed)
    start = int(length / 3)
    end = int(2 * length / 3)
    
    # Fallback nếu đoạn cắt quá ngắn
    if end - start < 512: 
        return y_trimmed
        
    return y_trimmed[start:end]

def visualize_segment_demo(root_dir):
    """Vẽ minh họa sự khác biệt giữa Strict Mode (Bài 1,2) và Full Mode (Bài 3)"""
    print("\n--- [DEMO] MINH HỌA PHÂN ĐOẠN TÍN HIỆU ---")
    sample_path = None
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.wav'): sample_path = os.path.join(root, f); break
        if sample_path: break
    if not sample_path: return

    y, sr = librosa.load(sample_path, sr=FS)
    y_trim, idx_trim = librosa.effects.trim(y, top_db=30)

    # Tính toán chỉ số cho 1/3 giữa (trên hệ trục gốc)
    start_trim = idx_trim[0]
    len_trim = idx_trim[1] - idx_trim[0]
    start_mid = start_trim + int(len_trim/3)
    end_mid = start_trim + int(2*len_trim/3)

    # Set seaborn style for modern look
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    time = np.arange(len(y)) / sr
    plt.plot(time, y, color='#4B8BBE', linewidth=2, label='Tín hiệu gốc')

    # Highlight HMM region
    plt.axvspan(time[start_trim], time[idx_trim[1]], color='#43AA8B', alpha=0.25, lw=0, label='Vùng HMM (Full Voiced)')
    plt.text((time[start_trim]+time[idx_trim[1]])/2, np.max(y)*0.8, 'HMM', color='#43AA8B', fontsize=14, ha='center', va='center', alpha=0.7, fontweight='bold')

    # Highlight FFT/K-means region
    plt.axvspan(time[start_mid], time[end_mid], color='#F76E11', alpha=0.35, lw=0, label='Vùng FFT/K-means (1/3 Giữa)')
    plt.text((time[start_mid]+time[end_mid])/2, np.max(y)*0.6, 'FFT/K-means', color='#F76E11', fontsize=14, ha='center', va='center', alpha=0.7, fontweight='bold')

    plt.title(f"Minh họa phân đoạn tín hiệu\n{os.path.basename(sample_path)}", fontsize=18, fontweight='bold', color='#22223B')
    plt.xlabel("Thời gian (s)", fontsize=14)
    plt.ylabel("Biên độ", fontsize=14)
    plt.legend(loc='upper right', fontsize=13, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'Demo_Segment_Selection.png'), dpi=120)
    plt.close()
    print("-> Đã lưu ảnh minh họa phân đoạn đẹp hơn.")

def load_data_paths(root_dir):
    data_paths = {v: [] for v in VOWELS}
    if not os.path.exists(root_dir): return data_paths
    for person in os.listdir(root_dir):
        path = os.path.join(root_dir, person)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.wav'):
                    v = file.split('.')[0].lower()
                    if v in VOWELS: data_paths[v].append(os.path.join(path, file))
    return data_paths

# --- CÁC HÀM TÍNH ĐẶC TRƯNG ---

def compute_fft_vector(file_path, n_fft):
    y, _ = librosa.load(file_path, sr=FS)
    # Bài 1 dùng strict_mode=True
    y_steady = extract_stable_segment(y, FS, strict_mode=True)
    
    # Padding nếu tín hiệu ngắn hơn n_fft
    if len(y_steady) < n_fft:
        y_steady = np.pad(y_steady, (0, n_fft - len(y_steady)))
        
    S = np.abs(librosa.stft(y_steady, n_fft=n_fft, hop_length=int(0.01*FS)))
    vec = np.mean(S, axis=1)
    return vec / (np.max(vec) + 1e-8)

def compute_mfcc_vectors(file_path, dim_type=13, n_fft=1024, strict=True):
    y, _ = librosa.load(file_path, sr=FS)
    y_steady = extract_stable_segment(y, FS, strict_mode=strict)
    
    mfcc = librosa.feature.mfcc(y=y_steady, sr=FS, n_mfcc=13, n_fft=n_fft, hop_length=int(0.01*FS))
    
    if dim_type == 13: return mfcc.T
    
    delta1 = librosa.feature.delta(mfcc)
    if dim_type == 26: return np.vstack([mfcc, delta1]).T
        
    delta2 = librosa.feature.delta(mfcc, order=2)
    if dim_type == 39: return np.vstack([mfcc, delta1, delta2]).T
    
    return mfcc.T

# --- HÀM VẼ ĐỒ THỊ ---

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=VOWELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VOWELS, yticklabels=VOWELS)
    plt.title(title); plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

def save_feature_plot(vectors_dict, title, xlabel, ylabel, filename, x_axis_values=None):
    plt.figure(figsize=(10, 6))
    first_key = list(vectors_dict.keys())[0]
    dim = len(vectors_dict[first_key])
    x_axis = np.arange(dim) if x_axis_values is None else x_axis_values
    
    markers = ['o', 'v', '^', 's', 'D']
    for i, v in enumerate(VOWELS):
        if v in vectors_dict:
            plt.plot(x_axis, vectors_dict[v], label=f'/{v}/', linewidth=1.5, alpha=0.8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

def save_convergence_plot(history_dict, title, filename):
    plt.figure(figsize=(10, 6))
    for v, history in history_dict.items():
        if history: plt.plot(history, label=f'/{v}/')
    plt.title(title); plt.xlabel("Iterations"); plt.ylabel("Log-Likelihood")
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

# ============================================================
# BÀI 1: FFT (Dùng Strict Mode)
# ============================================================
def run_task_1_fft(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 1: FFT (N_FFT: 512, 1024, 2048)")
    results = []
    n_fft_list = [512, 1024, 2048]
    best_acc, best_y_true, best_y_pred, best_n = 0, [], [], 0

    for n_fft in n_fft_list:
        models = {}
        for v in VOWELS:
            vecs = [compute_fft_vector(p, n_fft) for p in train_paths[v]]
            models[v] = np.mean(vecs, axis=0)
        # Vẽ với trục tần số Hz [0, 8000]
        freq_axis = np.linspace(0, FS/2, len(next(iter(models.values()))))
        save_feature_plot(models, f"Pho FFT Trung Binh (N={n_fft})", "Tan so (Hz)", "Bien do", 
                          f"Task1_Vectors_N{n_fft}.png", x_axis_values=freq_axis)

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
            best_acc = acc; best_n = n_fft; best_y_true, best_y_pred = y_true, y_pred
            best_models = models  # Lưu model tốt nhất

    # In bảng kết quả trực quan
    print("\nBẢNG KẾT QUẢ ACCURACY (FFT)")
    print("+---------+----------+")
    print("| N_FFT   | Accuracy |")
    print("+---------+----------+")
    for r in results:
        print(f"| {r['N_FFT']:<7} | {r['Accuracy']:<8.2f} |")
    print("+---------+----------+\n")

    # Lưu model tốt nhất
    with open(os.path.join(RESULT_DIR, 'Task1_Best_FFT_Model.pkl'), 'wb') as f:
        pickle.dump({'models': best_models, 'n_fft': best_n, 'accuracy': best_acc}, f)
    print(f"-> Đã lưu model FFT tốt nhất (N_FFT={best_n}, Acc={best_acc:.2f}%)")

    # Lưu file csv
    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task1_Results.csv'), index=False)
    save_confusion_matrix(best_y_true, best_y_pred, f"CM Bai 1 (Best N={best_n})", "Task1_Best_CM.png")

    # Vẽ bar plot trực quan kết quả accuracy
    df1 = pd.DataFrame(results)
    plt.figure(figsize=(7,5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df1, x='N_FFT', y='Accuracy', palette='Blues', edgecolor='black')
    plt.title("Accuracy theo N_FFT (FFT)", fontsize=15, fontweight='bold')
    plt.xlabel("N_FFT", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, color='black', fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'Task1_Accuracy_Bar.png'), dpi=120)
    plt.close()

# ============================================================
# BÀI 2: K-MEANS (Dùng Strict Mode)
# ============================================================
def run_task_2_kmeans(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 2: K-MEANS (K: 2, 3, 4, 5)")
    k_list = [2, 3, 4, 5]
    
    # Cache MFCC-13 với strict=True
    print("   [Cache] Extracting MFCC-13 (Strict Mode)...")
    train_feats = {v: [compute_mfcc_vectors(p, 13, strict=True) for p in train_paths[v]] for v in VOWELS}
    test_feats = {v: [compute_mfcc_vectors(p, 13, strict=True) for p in test_paths[v]] for v in VOWELS}

    # Vẽ vector trung bình toàn cục
    global_means = {v: np.mean(np.vstack(train_feats[v]), axis=0) for v in VOWELS}
    save_feature_plot(global_means, "Vector MFCC Trung Binh", "MFCC Index", "Value", "Task2_Global_MFCC.png")

    results = []
    best_acc, best_y_true, best_y_pred, best_k = 0, [], [], 0

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
            best_acc = acc; best_k = k; best_y_true, best_y_pred = y_true, y_pred
            best_codebooks = codebooks  # Lưu codebook tốt nhất

    # In bảng kết quả trực quan
    print("\nBẢNG KẾT QUẢ ACCURACY (K-means)")
    print("+-----+----------+")
    print("| K   | Accuracy |")
    print("+-----+----------+")
    for r in results:
        print(f"| {r['K']:<3} | {r['Accuracy']:<8.2f} |")
    print("+-----+----------+\n")

    # Lưu model tốt nhất
    with open(os.path.join(RESULT_DIR, 'Task2_Best_KMeans_Model.pkl'), 'wb') as f:
        pickle.dump({'codebooks': best_codebooks, 'k': best_k, 'accuracy': best_acc}, f)
    print(f"-> Đã lưu model K-means tốt nhất (K={best_k}, Acc={best_acc:.2f}%)")

    # Lưu file csv
    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task2_Results.csv'), index=False)
    save_confusion_matrix(best_y_true, best_y_pred, f"CM Bai 2 (Best K={best_k})", "Task2_Best_CM.png")

    # Vẽ bar plot trực quan kết quả accuracy
    df2 = pd.DataFrame(results)
    plt.figure(figsize=(7,5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df2, x='K', y='Accuracy', palette='Oranges', edgecolor='black')
    plt.title("Accuracy theo K (K-means)", fontsize=15, fontweight='bold')
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, color='black', fontweight='bold', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'Task2_Accuracy_Bar.png'), dpi=120)
    plt.close()

# ============================================================
# BÀI 3: HMM (Dùng Full Mode + Cross-Validation)
# ============================================================
def run_task_3_hmm(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 3: HMM (5-Fold Cross Validation)")
    
    dims = [13, 26, 39]; n_states = [3, 4, 5]; n_mixes = [1, 2, 3]
    
    # Cache features với strict=False (lấy toàn bộ)
    print("   [Cache] Extracting features (Full Mode for HMM)...")
    cache_train = {}; cache_test = {}
    
    for d in dims:
        cache_train[d] = {v: [compute_mfcc_vectors(p, d, strict=False) for p in train_paths[v]] for v in VOWELS}
        cache_test[d] = {v: [compute_mfcc_vectors(p, d, strict=False) for p in test_paths[v]] for v in VOWELS}

    best_cv_acc = -1; best_params = None
    results = []
    
    # K-Fold Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n   {'Config':<15} | {'CV-Acc':<8} | {'Status'}")
    print("-" * 45)

    for d in dims:
        for s in n_states:
            for m in n_mixes:
                config_str = f"D{d}_S{s}_M{m}"
                fold_accuracies = []
                
                # Thực hiện Cross-Validation
                # Do cấu trúc dữ liệu theo dict {vowel: [files]}, ta phải loop K-fold thủ công cho từng vowel
                # Nhưng để đơn giản, ta sẽ loop K-Fold trên index (0..N) và áp dụng cho tất cả vowel
                
                # Giả sử số lượng file mỗi nguyên âm bằng nhau hoặc xấp xỉ
                n_samples = len(cache_train[d]['a']) 
                indices = np.arange(n_samples)
                
                for train_idx, val_idx in kf.split(indices):
                    models = {}
                    valid_fold = True
                    
                    # Train phase (Fold k)
                    for v in VOWELS:
                        # Lấy dữ liệu theo index của fold
                        X_fold_train = [cache_train[d][v][i] for i in train_idx if i < len(cache_train[d][v])]
                        if not X_fold_train: continue
                        
                        lengths = [len(x) for x in X_fold_train]
                        X_concat = np.vstack(X_fold_train)
                        
                        try:
                            model = hmm.GMMHMM(n_components=s, n_mix=m, covariance_type='diag', 
                                               n_iter=20, random_state=42, min_covar=0.01, verbose=False)
                            model.fit(X_concat, lengths)
                            models[v] = model
                        except: valid_fold = False
                    
                    if not valid_fold or len(models) < 5: 
                        fold_accuracies.append(0); continue

                    # Validation phase (Fold k)
                    y_true_fold, y_pred_fold = [], []
                    for v_true in VOWELS:
                        X_fold_val = [cache_train[d][v_true][i] for i in val_idx if i < len(cache_train[d][v_true])]
                        for feat in X_fold_val:
                            best_s = -float('inf'); best_v = VOWELS[0]
                            for v_mod, mod in models.items():
                                try:
                                    score = mod.score(feat)
                                    if score > best_s: best_s = score; best_v = v_mod
                                except: pass
                            y_true_fold.append(v_true); y_pred_fold.append(best_v)
                    
                    fold_accuracies.append(accuracy_score(y_true_fold, y_pred_fold))
                
                avg_acc = np.mean(fold_accuracies) * 100
                results.append({'MFCC_Dim': d, 'States': s, 'Mixes': m, 'Accuracy': avg_acc, 'Config': config_str})
                print(f"   {config_str:<15} | {avg_acc:.2f}%   | 5-Fold CV")
                
                if avg_acc > best_cv_acc:
                    best_cv_acc = avg_acc; best_params = (d, s, m)

    # Lưu kết quả Grid Search
    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task3_GridSearch_CV.csv'), index=False)
    
    # Vẽ biểu đồ so sánh tham số (đẹp và rõ ràng hơn)
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_res, x='States', y='Accuracy', hue='MFCC_Dim', palette='Set2', edgecolor='black')
    plt.title("Độ chính xác trung bình (Cross-Validation)", fontsize=18, fontweight='bold', color='#22223B')
    plt.xlabel("Số trạng thái (States)", fontsize=14)
    plt.ylabel("Độ chính xác (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(title='MFCC Dim', fontsize=12, title_fontsize=13, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Thêm nhãn giá trị lên từng cột
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=11, color='black', fontweight='bold',
                        xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'Task3_CV_Analysis.png'), dpi=120)
    plt.close()

    print(f"\n   -> Best Params (CV): D={best_params[0]}, S={best_params[1]}, M={best_params[2]} (Acc={best_cv_acc:.2f}%)")
    
    # 3. RE-TRAIN & FINAL TEST
    print("   [Re-train] Training best model on FULL dataset...")
    d_best, s_best, m_best = best_params
    final_models = {}
    history_log = {}
    
    for v in VOWELS:
        X_list = cache_train[d_best][v]
        lengths = [len(x) for x in X_list]
        X_concat = np.vstack(X_list)
        
        model = hmm.GMMHMM(n_components=s_best, n_mix=m_best, covariance_type='diag', 
                           n_iter=50, random_state=42, min_covar=0.01, verbose=False, tol=0.001)
        model.fit(X_concat, lengths)
        final_models[v] = model
        history_log[v] = model.monitor_.history

    y_true, y_pred = [], []
    for v_true in VOWELS:
        for feat in cache_test[d_best][v_true]:
            best_s = -float('inf'); best_v = VOWELS[0]
            for v_mod, mod in final_models.items():
                try:
                    score = mod.score(feat)
                    if score > best_s: best_s = score; best_v = v_mod
                except: pass
            y_true.append(v_true); y_pred.append(best_v)
            
    final_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n   ===> FINAL TEST ACCURACY: {final_acc:.2f}% <===")
    
    # Lưu model HMM tốt nhất
    with open(os.path.join(RESULT_DIR, 'Task3_Best_HMM_Model.pkl'), 'wb') as f:
        pickle.dump({
            'models': final_models, 
            'params': best_params,
            'accuracy': final_acc,
            'mfcc_dim': d_best,
            'n_states': s_best,
            'n_mixes': m_best
        }, f)
    print(f"-> Đã lưu model HMM tốt nhất (D={d_best}_S={s_best}_M={m_best}, Acc={final_acc:.2f}%)")
    
    config_name = f"D{d_best}_S{s_best}_M{m_best}"
    save_confusion_matrix(y_true, y_pred, f"Final CM ({config_name})", "Task3_Final_CM.png")
    save_convergence_plot(history_log, f"Hoi tu Log-Likelihood ({config_name})", "Task3_Convergence.png")

def demo_load_models():
    """Demo hàm load và sử dụng các model đã lưu"""
    print("\n" + "="*40)
    print("DEMO: LOAD VÀ SỬ DỤNG CÁC MODEL ĐÃ LƯU")
    
    # Load FFT model
    try:
        with open(os.path.join(RESULT_DIR, 'Task1_Best_FFT_Model.pkl'), 'rb') as f:
            fft_model = pickle.load(f)
        print(f"✓ FFT Model: N_FFT={fft_model['n_fft']}, Accuracy={fft_model['accuracy']:.2f}%")
    except: print("✗ Không tìm thấy FFT model")
    
    # Load K-means model  
    try:
        with open(os.path.join(RESULT_DIR, 'Task2_Best_KMeans_Model.pkl'), 'rb') as f:
            kmeans_model = pickle.load(f)
        print(f"✓ K-means Model: K={kmeans_model['k']}, Accuracy={kmeans_model['accuracy']:.2f}%")
    except: print("✗ Không tìm thấy K-means model")
    
    # Load HMM model
    try:
        with open(os.path.join(RESULT_DIR, 'Task3_Best_HMM_Model.pkl'), 'rb') as f:
            hmm_model = pickle.load(f)
        print(f"✓ HMM Model: D={hmm_model['mfcc_dim']}_S={hmm_model['n_states']}_M={hmm_model['n_mixes']}, Accuracy={hmm_model['accuracy']:.2f}%")
    except: print("✗ Không tìm thấy HMM model")

# --- MAIN ---
if __name__ == "__main__":
    train_paths = load_data_paths(TRAIN_DIR)
    test_paths = load_data_paths(TEST_DIR)
    
    if any(len(x) > 0 for x in train_paths.values()):
        visualize_segment_demo(TRAIN_DIR)
        run_task_1_fft(train_paths, test_paths)
        run_task_2_kmeans(train_paths, test_paths)
        run_task_3_hmm(train_paths, test_paths)
        demo_load_models()  # Demo load các model đã lưu
        print(f"\n[DONE] Kết quả đã lưu tại thư mục '{RESULT_DIR}'.")
    else:
        print("Lỗi: Không tìm thấy dữ liệu wav.")