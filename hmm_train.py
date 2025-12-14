import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from hmmlearn import hmm
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

# Task 3 runner

def run_task_3_hmm(train_paths, test_paths):
    print("\n" + "="*40)
    print("BÀI 3: HMM (5-Fold Cross Validation)")

    dims = [13, 26, 39]; n_states = [3, 4, 5]; n_mixes = [1, 2, 3]

    print("   [Cache] Extracting features (Full Mode for HMM)...")
    cache_train = {}; cache_test = {}
    for d in dims:
        cache_train[d] = {v: [compute_mfcc_vectors(p, d, strict=False) for p in train_paths[v]] for v in VOWELS}
        cache_test[d] = {v: [compute_mfcc_vectors(p, d, strict=False) for p in test_paths[v]] for v in VOWELS}

    best_cv_acc = -1; best_params = None
    results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n   {'Config':<15} | {'CV-Acc':<8} | {'Status'}")
    print("-" * 45)

    for d in dims:
        for s in n_states:
            for m in n_mixes:
                config_str = f"D{d}_S{s}_M{m}"
                fold_accuracies = []
                n_samples = len(cache_train[d]['a'])
                indices = np.arange(n_samples)
                for train_idx, val_idx in kf.split(indices):
                    models = {}
                    valid_fold = True
                    for v in VOWELS:
                        X_fold_train = [cache_train[d][v][i] for i in train_idx if i < len(cache_train[d][v])]
                        if not X_fold_train:
                            continue
                        lengths = [len(x) for x in X_fold_train]
                        X_concat = np.vstack(X_fold_train)
                        try:
                            model = hmm.GMMHMM(n_components=s, n_mix=m, covariance_type='diag', n_iter=20, random_state=42, min_covar=0.01, verbose=False)
                            model.fit(X_concat, lengths)
                            models[v] = model
                        except:
                            valid_fold = False
                    if not valid_fold or len(models) < 5:
                        fold_accuracies.append(0); continue
                    y_true_fold, y_pred_fold = [], []
                    for v_true in VOWELS:
                        X_fold_val = [cache_train[d][v_true][i] for i in val_idx if i < len(cache_train[d][v_true])]
                        for feat in X_fold_val:
                            best_s = -float('inf'); best_v = VOWELS[0]
                            for v_mod, mod in models.items():
                                try:
                                    score = mod.score(feat)
                                    if score > best_s: best_s = score; best_v = v_mod
                                except:
                                    pass
                            y_true_fold.append(v_true); y_pred_fold.append(best_v)
                    fold_accuracies.append(accuracy_score(y_true_fold, y_pred_fold))
                avg_acc = np.mean(fold_accuracies) * 100
                results.append({'MFCC_Dim': d, 'States': s, 'Mixes': m, 'Accuracy': avg_acc, 'Config': config_str})
                print(f"   {config_str:<15} | {avg_acc:.2f}%   | 5-Fold CV")
                if avg_acc > best_cv_acc:
                    best_cv_acc = avg_acc; best_params = (d, s, m)

    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, 'Task3_GridSearch_CV.csv'), index=False)

    df_res = pd.DataFrame(results)
    plt.figure(figsize=(12, 7)); sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_res, x='States', y='Accuracy', hue='MFCC_Dim', palette='Set2', edgecolor='black')
    plt.title("Độ chính xác trung bình (Cross-Validation)", fontsize=18, fontweight='bold')
    plt.xlabel("Số trạng thái (States)", fontsize=14); plt.ylabel("Độ chính xác (%)", fontsize=14); plt.ylim(0,100)
    plt.legend(title='MFCC Dim', fontsize=12, title_fontsize=13, loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    for p in ax.patches:
        h = p.get_height();
        if not np.isnan(h): ax.annotate(f"{h:.1f}", (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0,3), textcoords='offset points')
    plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR, 'Task3_CV_Analysis.png'), dpi=120); plt.close()

    print(f"\n   -> Best Params (CV): D={best_params[0]}, S={best_params[1]}, M={best_params[2]} (Acc={best_cv_acc:.2f}%)")

    print("   [Re-train] Training best model on FULL dataset...")
    d_best, s_best, m_best = best_params
    final_models = {}; history_log = {}
    for v in VOWELS:
        X_list = cache_train[d_best][v]
        lengths = [len(x) for x in X_list]
        X_concat = np.vstack(X_list)
        model = hmm.GMMHMM(n_components=s_best, n_mix=m_best, covariance_type='diag', n_iter=50, random_state=42, min_covar=0.01, verbose=False, tol=0.001)
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
                except:
                    pass
            y_true.append(v_true); y_pred.append(best_v)
    final_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n   ===> FINAL TEST ACCURACY: {final_acc:.2f}% <===")

    with open(os.path.join(RESULT_DIR, 'Task3_Best_HMM_Model.pkl'), 'wb') as f:
        pickle.dump({'models': final_models, 'params': best_params, 'accuracy': final_acc, 'mfcc_dim': d_best, 'n_states': s_best, 'n_mixes': m_best}, f)
    print(f"-> Đã lưu model HMM tốt nhất (D={d_best}_S={s_best}_M={m_best}, Acc={final_acc:.2f}%)")

    # Optional: convergence plot
    plt.figure(figsize=(10,6))
    for v, history in history_log.items():
        if history: plt.plot(history, label=f'/{v}/')
    plt.title(f"Hội tụ Log-Likelihood (D{d_best}_S{s_best}_M{m_best})")
    plt.xlabel("Iterations"); plt.ylabel("Log-Likelihood"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'Task3_Convergence.png')); plt.close()

if __name__ == '__main__':
    train_paths = load_data_paths(TRAIN_DIR)
    test_paths = load_data_paths(TEST_DIR)
    if any(len(x) > 0 for x in train_paths.values()):
        run_task_3_hmm(train_paths, test_paths)
        print(f"\n[DONE] Kết quả đã lưu tại thư mục '{RESULT_DIR}'.")
    else:
        print("Lỗi: Không tìm thấy dữ liệu wav.")
