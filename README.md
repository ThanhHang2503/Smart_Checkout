
# Nhận Diện Khuôn Mặt Thông Minh với LFW Dataset

## 🌟 Giới Thiệu

Dự án này hướng đến việc **xây dựng và so sánh hai mô hình học sâu** cho bài toán nhận diện khuôn mặt: **FaceNet (Triplet Loss)** và **ResNet18 + ArcFace (Additive Angular Margin Loss)**. Mục tiêu chính là xác định mô hình phù hợp hơn cho các ứng dụng xác thực thông minh trong điều kiện thực tế (ví dụ: chấm công, bảo mật, nhận diện khách hàng...).

Bộ dữ liệu sử dụng: **LFW (Labeled Faces in the Wild)** – tập ảnh khuôn mặt trong môi trường không kiểm soát, có sự đa dạng về biểu cảm, ánh sáng và góc nhìn.

## 🧠 Các Thành Phần Chính

### 1. 📊 Phân Tích Dữ Liệu (EDA)
- `lfw_analysis.py`: Phân tích tổng quan bộ dữ liệu LFW, tạo biểu đồ số lượng ảnh theo từng người, phân bố dữ liệu.
- `eda.py`, `edaTrain.py`: Phân tích đặc trưng tập cặp ảnh khớp và không khớp (matching/mismatching pairs).

### 2. 🛠️ Chuẩn Bị Dữ Liệu
- `dataset_splitter.py`: Tách dữ liệu thành tập huấn luyện, kiểm định và kiểm thử. Cung cấp `FaceDataset` và `DataLoader`.

### 3. 🏋️ Huấn Luyện Mô Hình
- `train_facenet.py`: Huấn luyện FaceNet (InceptionResNetV1 tiền huấn luyện VGGFace2), sử dụng Triplet Loss.
- `train_resnet18.py`: Huấn luyện ResNet18 (pretrained ImageNet) với ArcFace Loss, tối ưu phân biệt góc giữa các vector embedding.

### 4. 📈 Đánh Giá Hiệu Suất
- `evaluate.py`, `evalutateVer2.py`:
  - Tính toán embedding từ các cặp ảnh.
  - Đo khoảng cách Euclidean / cosine giữa embedding.
  - Đánh giá Accuracy, Precision, Recall, F1-score, AUC.
  - Vẽ ROC curve, confusion matrix, biểu đồ phân phối khoảng cách.

### 5. 👁️ Nhận Diện Thời Gian Thực
- `Face_Indentification.py`:
  - Dùng webcam để phát hiện và nhận diện khuôn mặt.
  - Trích xuất embedding qua `insightface`.
  - So sánh với ảnh tham chiếu bằng cosine similarity.

### 6. 🧭 Sơ Đồ Quy Trình
- `pipeline_diagram.py`: Sinh sơ đồ `pipeline_overview.png` mô tả toàn bộ luồng xử lý của hệ thống.

## ⚙️ Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python
- **Học sâu:** PyTorch, facenet-pytorch, torchvision
- **Tiền xử lý & trực quan hóa:** OpenCV, NumPy, Pandas, Matplotlib, Seaborn
- **Real-time inference:** insightface
- **Sơ đồ quy trình:** graphviz

## 🚀 Hướng Dẫn Cài Đặt

1. **Clone dự án**
```bash
git clone https://github.com/ThanhHang2503/Smart_Checkout
cd Smart_Checkout
```

2. **Cài thư viện phụ thuộc**
```bash
pip install -r requirements.txt
```

> ⚠️ *`insightface` và `graphviz` cần thêm các bước cài đặt hệ thống. Tham khảo tài liệu chính thức.*

3. **Chuẩn bị dữ liệu**
- Tải tập dữ liệu `lfw-deepfunneled` và các tệp CSV.
- Link dataset: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- Cập nhật đường dẫn trong các tệp `.py`.
- Chạy:
```bash
python dataset_splitter.py
```

4. **Huấn luyện mô hình**
```bash
python train_facenet.py
python train_resnet18.py
```

5. **Đánh giá mô hình**
```bash
python evaluate.py
python evalutateVer2.py
```

6. **Chạy demo nhận diện webcam**
```bash
python Face_Indentification.py
```
- Nhấn `s` để lưu khuôn mặt tham chiếu.
- Nhấn `q` để thoát.

## 🗂️ Cấu Trúc Thư Mục

```
Smart_Checkout/
├── models/
│   ├── facenet.pth
│   └── resnet18_arcface.pth
├── results/
│   ├── roc_comparison.png
│   ├── confusion_matrix_*.png
│   ├── distance_distribution_*.png
│   └── metrics_comparison.png
├── *.py (các tập tin code)
├── requirements.txt
├── README.md
└── *.png (biểu đồ và sơ đồ)
```

## ✅ Kết Quả Mong Đợi

- Mô hình được huấn luyện và lưu tại `models/`.
- Biểu đồ và báo cáo đánh giá chi tiết tại `results/`.
- Ứng dụng demo real-time có thể nhận diện khuôn mặt qua webcam.