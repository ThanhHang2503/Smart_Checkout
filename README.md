Giới thiệu
\n
Dự án này sử dụng các mô hình học sâu như FaceNet và ArcFace+ResNet để phát triển hệ thống nhận diện khuôn mặt, nhằm ứng dụng trong các bài toán nhận dạng và xác thực khuôn mặt. Dự án tập trung vào việc xử lý dữ liệu khuôn mặt trong các môi trường không kiểm soát và sử dụng các thuật toán tiên tiến để cải thiện độ chính xác và khả năng phân biệt khuôn mặt.

Các mô hình sử dụng:
FaceNet (Triplet Loss): Sử dụng mô hình học sâu để tạo ra các vector đặc trưng cho khuôn mặt. FaceNet giúp chuyển đổi khuôn mặt thành các vector có thể dễ dàng so sánh với nhau bằng khoảng cách Euclidean.

ArcFace + ResNet: Sử dụng mô hình ArcFace, với cải tiến margin góc để phân biệt khuôn mặt tốt hơn, kết hợp với mạng ResNet để nâng cao hiệu suất nhận diện khuôn mặt.

Công nghệ sử dụng
Python: Ngôn ngữ lập trình chính để phát triển và triển khai hệ thống.

TensorFlow/Keras/PyTorch: Thư viện học sâu cho việc xây dựng và huấn luyện mô hình.

OpenCV: Thư viện xử lý ảnh để trích xuất và phát hiện khuôn mặt.

Cài đặt và sử dụng: 
B1: Clone dự án về máy: 
git clone https://github.com/username/project-name.git
B2: Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
B3: Tiến hành huấn luyện mô hình (nếu chưa có mô hình đã huấn luyện)
B4: Sử dụng mô hình đã huấn luyện để nhận diện khuôn mặt
