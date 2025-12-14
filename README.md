# Nhận dạng nguyên âm — Huấn luyện & Demo

Kho dự án gồm 3 bài toán nhận dạng nguyên âm tiếng Việt:
- Bài 1: Phân loại dựa trên FFT
- Bài 2: K-Means với đặc trưng MFCC
- Bài 3: HMM với MFCC (+ delta) và Cross-Validation 5-fold

Các mô hình tốt nhất của mỗi bài toán được lưu trong thư mục `results/`. Script demo sẽ nạp mô hình và dự đoán trên một file kiểm thử mẫu.

## Cấu trúc thư mục
```
ck/
├─ train.py                 # Script chính: huấn luyện + vẽ biểu đồ + lưu model tốt nhất
├─ demo_model_results.py   # Demo: nạp model trong results và dự đoán trên 1 file wav mẫu
├─ requirements.txt        # Thư viện phụ thuộc
├─ results/                # Kết quả: CSV, biểu đồ, file model (.pkl)
└─ signals/
   ├─ NguyenAmHuanLuyen-16k/   # TRAIN_DIR — dữ liệu huấn luyện
   └─ NguyenAmKiemThu-16k/     # TEST_DIR — dữ liệu kiểm thử
      └─ 25MLM/                # Thư mục mẫu cho demo (chứa các file wav)
```

## Yêu cầu môi trường
- Python 3.9 trở lên (khuyến nghị)
- Windows PowerShell (các lệnh bên dưới dùng cú pháp PowerShell)
- Bộ dữ liệu âm thanh sắp xếp đúng cấu trúc thư mục như trên

## Cài đặt thư viện
Chạy từ thư mục gốc dự án (`ck/`):

```powershell
pip install -r requirements.txt
```

Các thư viện sử dụng (trong `requirements.txt`):
- numpy, pandas
- matplotlib, seaborn
- librosa
- scikit-learn
- hmmlearn

## Chạy huấn luyện + tạo báo cáo (main)
Huấn luyện cả 3 bài toán trên `TRAIN_DIR` và kiểm thử trên `TEST_DIR`. Tự động tạo biểu đồ/CSV và lưu mô hình tốt nhất.

```powershell
python -u "train.py"
```

Sau khi chạy, thư mục `results/` sẽ có:
- Bài 1: `Task1_Results.csv`, `Task1_Vectors_N*.png`, `Task1_Best_CM.png`, `Task1_Accuracy_Bar.png`, `Task1_Best_FFT_Model.pkl`
- Bài 2: `Task2_Results.csv`, `Task2_Global_MFCC.png`, `Task2_Best_CM.png`, `Task2_Accuracy_Bar.png`, `Task2_Best_KMeans_Model.pkl`
- Bài 3: `Task3_GridSearch_CV.csv`, `Task3_CV_Analysis.png`, `Task3_Final_CM.png`, `Task3_Convergence.png`, `Task3_Best_HMM_Model.pkl`
- Minh họa cắt tín hiệu: `Demo_Segment_Selection.png`

## Chạy demo (dự đoán trên 1 file wav mẫu)
Sau khi huấn luyện xong, chạy demo để nạp các mô hình đã lưu và dự đoán trên một file trong `signals/NguyenAmKiemThu-16k/25MLM`.

```powershell
python -u "demo_model_results.py"
```

Demo sẽ in ra nguyên âm dự đoán theo từng mô hình:
- FFT (dựa khoảng cách vector đặc trưng)
- K-Means (khoảng cách trung bình đến tâm cụm)
- HMM (điểm log-likelihood)

## Ghi chú
- `train.py` tự động lưu mô hình tốt nhất (định dạng pickle) vào `results/`.
- Đảm bảo dữ liệu `wav` nằm đúng trong `signals/` và các thư mục con như cấu trúc đã nêu.
- Nếu thiếu file mô hình `.pkl`, hãy chạy lại script huấn luyện trước khi chạy demo.
