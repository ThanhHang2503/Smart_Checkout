from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Pipeline Xử Lý Dữ Liệu và Huấn Luyện Mô Hình', graph_attr={'rankdir': 'TB'})
dot.attr(fontsize='20')

# Define node attributes
node_attrs = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue', 'fontname': 'Arial'}

# --- Main Pipeline Nodes ---
dot.node('A', 'Bước 1: Tiền xử lý & Trích xuất đặc trưng', **node_attrs)

# --- Subgraph for FaceNet ---
with dot.subgraph(name='cluster_facenet') as c_facenet:
    c_facenet.node('B1', 'Bước 1.1: Trích xuất đặc trưng FaceNet\n(128 chiều)', **node_attrs)
    c_facenet.node('C1', 'Bước 2: Huấn luyện mô hình & So sánh Embedding\nHuấn luyện FaceNet + Triplet Loss', **node_attrs)
    c_facenet.node('D1', 'Bước 3: Đánh giá mô hình & phân tích chỉ số\nĐánh giá FaceNet + Triplet Loss', **node_attrs)
    c_facenet.edge('B1', 'C1')
    c_facenet.edge('C1', 'D1')

# --- Subgraph for ResNet18 + ArcFace ---
with dot.subgraph(name='cluster_resnet_arcface') as c_resnet:
    c_resnet.node('B2', 'Bước 1.1: Trích xuất đặc trưng ResNet18 + ArcFace\n(512 chiều)', **node_attrs)
    c_resnet.node('C2', 'Bước 2: Huấn luyện mô hình & So sánh Embedding\nHuấn luyện ResNet18 + ArcFace', **node_attrs)
    c_resnet.node('D2', 'Bước 3: Đánh giá mô hình & phân tích chỉ số\nĐánh giá ResNet18 + ArcFace', **node_attrs)
    c_resnet.edge('B2', 'C2')
    c_resnet.edge('C2', 'D2')

# --- Final Step Node ---
dot.node('E', 'Bước 4: Phân tích kết quả & đề xuất mô hình phù hợp', **node_attrs)

# --- Edges connecting main steps and subgraphs ---
dot.edge('A', 'B1', lhead='cluster_facenet')
dot.edge('A', 'B2', lhead='cluster_resnet_arcface')
dot.edge('D1', 'E', ltail='cluster_facenet')
dot.edge('D2', 'E', ltail='cluster_resnet_arcface')


# --- Detailed descriptions (can be added as tooltips or separate text if needed, keeping diagram clean) ---
# Descriptions from the user query:
# A: "Bộ dữ liệu: LFW (13,000+ ảnh khuôn mặt trong điều kiện thực tế).\nTiền xử lý bằng MTCNN:\nPhát hiện và căn chỉnh khuôn mặt.\nCắt và chuẩn hóa ảnh về kích thước chuẩn (160x160)."
# B1: "FaceNet: ánh xạ ảnh khuôn mặt vào không gian nhúng 128 chiều."
# B2: "ResNet18 + ArcFace: tạo vector đặc trưng 512 chiều tối ưu theo góc (angular margin)."
# C1: "FaceNet + Triplet Loss: học sao cho khoảng cách giữa các vector khuôn mặt giống nhau nhỏ hơn so với khác người."
# C2: "ResNet18 + ArcFace: tối ưu theo khoảng cách góc, tạo phân biệt rõ ràng hơn giữa các cá nhân."
# D1 & D2 (Shared concepts, then model specific):
# Shared_Eval: "Tiêu chí đánh giá: Accuracy, Precision, Recall, F1-score, AUC.\nPhép thử thực nghiệm: Dựa trên 500 cặp \"cùng danh tính\" và 500 cặp \"khác danh tính\".\nDự đoán nhãn và so sánh với ground truth → tạo ma trận nhầm lẫn.\nPhân tích embedding distribution để giải thích kết quả."
# E: "FaceNet:\nƯu điểm: cân bằng Precision & Recall.\nNhược điểm: chưa phân biệt tốt embedding → hiệu suất tổng thể thấp.\nResNet18 + ArcFace:\nƯu điểm: Recall và F1-score cao (tốt với các hệ thống yêu cầu nhạy).\nNhược điểm: dễ nhầm lẫn cặp khác danh tính → Precision thấp."

# Render and save the diagram
try:
    dot.render('pipeline_overview', view=False, format='png', cleanup=True)
    print("Sơ đồ pipeline đã được tạo và lưu vào file 'pipeline_overview.png'")
except Exception as e:
    print(f"Đã xảy ra lỗi khi tạo sơ đồ: {e}")
    print("Vui lòng đảm bảo bạn đã cài đặt Graphviz và thêm nó vào PATH của hệ thống.")
    print("Bạn có thể tải Graphviz từ: https://graphviz.org/download/") 