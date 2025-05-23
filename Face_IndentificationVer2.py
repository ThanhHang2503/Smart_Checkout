import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from facenet_pytorch import InceptionResnetV1
from datetime import datetime
from numpy.linalg import norm
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading # Để chạy các tác vụ nặng không làm đơ GUI

# Đường dẫn tới các file mô hình
model_dir = "models"
facenet_model_path = os.path.join(model_dir, "facenet.pth")
resnet18_model_path = os.path.join(model_dir, "resnet18_arcface.pth")
reference_faces_path = "reference_faces.pkl"

# Thiết lập thiết bị và cấu hình CUDA
print(f"PyTorch version: {torch.__version__}")

# Kiểm tra CUDA
if torch.cuda.is_available():
    # Đặt các tùy chọn để tối ưu hiệu suất CUDA
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    device = torch.device("cuda")
    torch.cuda.empty_cache()  # Xóa bộ nhớ GPU đang được sử dụng trước đó
    
    print(f"Đang sử dụng: {device}")
    print(f"GPU đang sử dụng: {torch.cuda.get_device_name(0)}")
    print(f"Số lượng GPU khả dụng: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"Bộ nhớ GPU tối đa: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
    
    # Đặt thiết bị mặc định cho tensors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    print(f"Đang sử dụng: {device}")
    print("Không tìm thấy GPU. Đang sử dụng CPU.")
    print("Kiểm tra xem CUDA đã được cài đặt đúng cách chưa.")

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Định nghĩa mô hình FaceNet
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model.logits = nn.Linear(1792, embedding_size)
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Định nghĩa mô hình ResNet18+ArcFace
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=128, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, embeddings, labels):
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        W = nn.functional.normalize(self.W, p=2, dim=0)
        logits = torch.matmul(embeddings, W)
        
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        output *= self.s
        
        return output

class ResNet18ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size=128):
        super(ResNet18ArcFace, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
        self.arcface = ArcFaceLoss(num_classes, embedding_size)
    
    def forward(self, x, labels=None):
        embeddings = self.model(x)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return embeddings, logits
        return embeddings

# Tải mô hình
def load_models():
    # Đảm bảo sử dụng GPU nếu khả dụng
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Xóa bộ nhớ GPU đang được sử dụng trước đó
    
    # Tải FaceNet
    facenet_model = FaceNet(embedding_size=128)
    try:
        # Thử tải mô hình trực tiếp vào GPU
        if torch.cuda.is_available():
            facenet_model.load_state_dict(torch.load(facenet_model_path))
        else:
            facenet_model.load_state_dict(torch.load(facenet_model_path, map_location=device))
        print("Đã tải mô hình FaceNet thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình FaceNet: {e}")
        print("Thử phương pháp tải thay thế...")
    facenet_model = facenet_model.to(device)
    facenet_model.eval()
      # Tải ResNet18+ArcFace
    num_classes = 2826  # Số danh tính trong tập Train
    resnet18_model = ResNet18ArcFace(num_classes=num_classes, embedding_size=128)
    try:
        # Thử tải mô hình trực tiếp vào GPU
        if torch.cuda.is_available():
            resnet18_model.load_state_dict(torch.load(resnet18_model_path))
        else:
            resnet18_model.load_state_dict(torch.load(resnet18_model_path, map_location=device))
        print("Đã tải mô hình ResNet18+ArcFace thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình ResNet18+ArcFace: {e}")
        print("Thử tải với map_location=device...")
        resnet18_model.load_state_dict(torch.load(resnet18_model_path, map_location=device))
    
    resnet18_model = resnet18_model.to(device)
    resnet18_model.eval()
    
    # Kiểm tra xem mô hình đã được đưa lên GPU chưa
    if torch.cuda.is_available():
        print(f"FaceNet trên thiết bị: {next(facenet_model.parameters()).device}")
        print(f"ResNet18 trên thiết bị: {next(resnet18_model.parameters()).device}")
    
    return facenet_model, resnet18_model

# Hàm tính độ tương đồng cosine
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Hàm lấy embedding từ ảnh
def get_embedding(image, model):
    with torch.no_grad():
        # Trước tiên chuyển ảnh thành tensor
        img = transform(image)
        
        # Đưa tensor lên GPU
        if torch.cuda.is_available():
            # Theo dõi việc sử dụng bộ nhớ
            before_memory = torch.cuda.memory_allocated()
            
            # Thêm batch dimension và đưa lên GPU
            img = img.unsqueeze(0).to(device)
            
            # Tính toán embedding trên GPU
            embedding = model(img)
            
            # Chuyển embedding về CPU
            embedding_cpu = embedding.cpu().numpy()[0]
            
            # Giải phóng bộ nhớ GPU
            del img, embedding
            torch.cuda.empty_cache()
            
            after_memory = torch.cuda.memory_allocated()
            # Số bộ nhớ được sử dụng (trong bytes)
            memory_used = after_memory - before_memory
            
            return embedding_cpu
        else:
            # Nếu không có GPU, chạy trên CPU
            img = img.unsqueeze(0)
            embedding = model(img)
            return embedding.numpy()[0]

# Hàm phát hiện khuôn mặt sử dụng OpenCV
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Hàm lưu khuôn mặt tham chiếu
def save_reference_faces(reference_faces):
    with open(reference_faces_path, 'wb') as f:
        pickle.dump(reference_faces, f)
        
# Hàm tải khuôn mặt tham chiếu
def load_reference_faces():
    if os.path.exists(reference_faces_path):
        with open(reference_faces_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Lớp ứng dụng GUI
class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hệ Thống Nhận Diện Khuôn Mặt")
        # self.geometry("450x450") # Kích thước cửa sổ (tăng chiều cao để có chỗ cho widget mới)

        # Đặt kích thước cửa sổ ban đầu
        window_width = 450
        window_height = 450

        # Lấy kích thước màn hình
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Tính toán vị trí để căn giữa cửa sổ
        position_x = (screen_width // 2) - (window_width // 2)
        position_y = (screen_height // 2) - (window_height // 2)

        # Đặt kích thước và vị trí cho cửa sổ
        self.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

        self.facenet_model = None
        self.resnet18_model = None

        # Biến lưu trữ giá trị ngưỡng từ thanh trượt
        self.facenet_threshold_var = tk.DoubleVar(value=0.6)
        self.resnet_threshold_var = tk.DoubleVar(value=0.6)

        # Khung chính
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Tiêu đề
        title_label = ttk.Label(main_frame, text="So Sánh FaceNet và ResNet18+ArcFace", font=("Arial", 16, "bold"), anchor="center")
        title_label.pack(pady=10, fill=tk.X)

        # Khung cho các nút bấm
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)

        self.btn_add_face = ttk.Button(btn_frame, text="Thêm Khuôn Mặt Mới", command=self.add_face_thread_start, width=30)
        self.btn_add_face.grid(row=0, column=0, padx=10, pady=5)

        self.btn_scan_face = ttk.Button(btn_frame, text="Quét Khuôn Mặt", command=self.scan_face_thread_start, width=30)
        self.btn_scan_face.grid(row=1, column=0, padx=10, pady=5)
        
        self.btn_exit = ttk.Button(btn_frame, text="Thoát", command=self.quit_app, width=30)
        self.btn_exit.grid(row=2, column=0, padx=10, pady=5)

        # Khung cho cài đặt ngưỡng
        threshold_frame = ttk.LabelFrame(main_frame, text="Cài đặt Ngưỡng Nhận Diện", padding="10")
        threshold_frame.pack(pady=10, fill=tk.X)

        # Ngưỡng FaceNet
        facenet_label = ttk.Label(threshold_frame, text="FaceNet Threshold:")
        facenet_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.facenet_scale_label = ttk.Label(threshold_frame, text=f"{self.facenet_threshold_var.get():.2f}")
        self.facenet_scale_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        facenet_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.facenet_threshold_var, command=lambda val: self.facenet_scale_label.config(text=f"{float(val):.2f}"))
        facenet_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Ngưỡng ResNet18
        resnet_label = ttk.Label(threshold_frame, text="ResNet18 Threshold:")
        resnet_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.resnet_scale_label = ttk.Label(threshold_frame, text=f"{self.resnet_threshold_var.get():.2f}")
        self.resnet_scale_label.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        resnet_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.resnet_threshold_var, command=lambda val: self.resnet_scale_label.config(text=f"{float(val):.2f}"))
        resnet_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        threshold_frame.columnconfigure(1, weight=1) # Cho phép thanh trượt mở rộng

        # Thanh trạng thái (đơn giản)
        self.status_label = ttk.Label(main_frame, text="Đang khởi tạo...", anchor="center")
        self.status_label.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

        # Tải mô hình trong một luồng riêng để không làm đơ GUI
        self.load_models_thread()

    def load_models_thread(self):
        threading.Thread(target=self._load_models_task, daemon=True).start()

    def _load_models_task(self):
        self.status_label.config(text="Đang tải mô hình FaceNet và ResNet18...")
        self.btn_add_face.config(state=tk.DISABLED)
        self.btn_scan_face.config(state=tk.DISABLED)
        try:
            # Sử dụng hàm load_models toàn cục
            self.facenet_model, self.resnet18_model = load_models()
            self.status_label.config(text="Tải mô hình thành công!")
            # messagebox nên được gọi từ luồng chính nếu có thể, hoặc dùng self.after
            self.after(0, lambda: messagebox.showinfo("Thông báo", "Tải mô hình FaceNet và ResNet18 thành công!", parent=self))
            self.btn_add_face.config(state=tk.NORMAL)
            self.btn_scan_face.config(state=tk.NORMAL)
        except Exception as e:
            error_msg = f"Lỗi khi tải mô hình: {e}"
            self.status_label.config(text=error_msg)
            self.after(0, lambda: messagebox.showerror("Lỗi tải mô hình", error_msg, parent=self))
            # Cân nhắc việc đóng ứng dụng hoặc vô hiệu hóa các nút vĩnh viễn
            # self.destroy() 

    def add_face_thread_start(self):
        if not (self.facenet_model and self.resnet18_model):
            messagebox.showerror("Lỗi", "Mô hình chưa được tải xong hoặc có lỗi.", parent=self)
            return

        face_name = simpledialog.askstring("Nhập tên", "Nhập tên cho khuôn mặt:", parent=self)
        if face_name: # Nếu người dùng nhập tên và không nhấn cancel
            threading.Thread(target=self._capture_and_save_new_face, args=(face_name,), daemon=True).start()
        else:
            messagebox.showinfo("Thông báo", "Đã hủy thêm khuôn mặt.", parent=self)


    def _capture_and_save_new_face(self, face_name):
        reference_faces = load_reference_faces() # Hàm toàn cục
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror("Lỗi Camera", "Không thể mở camera.", parent=self))
            return

        face_added = False
        add_face_window_name = f"Them Khuon Mat: {face_name} - Nhan \'s\' de luu, \'q\' de thoat"
        cv2.namedWindow(add_face_window_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                self.after(0, lambda: messagebox.showerror("Lỗi Camera", "Không thể đọc frame từ camera.", parent=self))
                break
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Ten: {face_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Nhin thang vao camera va nhan \'s\' de luu.", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, "Nhan \'q\' de thoat.", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
            faces = detect_faces(frame) # Hàm toàn cục
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow(add_face_window_name, display_frame)
            key = cv2.waitKey(1)
            
            if key == ord('s') and len(faces) > 0:
                x_f, y_f, w_f, h_f = faces[0]
                face_img = frame[y_f:y_f+h_f, x_f:x_f+w_f]
                if face_img.size > 0:
                    try:
                        facenet_emb = get_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), self.facenet_model)
                        resnet_emb = get_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), self.resnet18_model)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        reference_faces[face_name] = {
                            'facenet_embedding': facenet_emb,
                            'resnet_embedding': resnet_emb,
                            'timestamp': timestamp
                        }
                        
                        os.makedirs("reference_images", exist_ok=True)
                        img_path = f"reference_images/{face_name}_{timestamp}.jpg"
                        cv2.imwrite(img_path, face_img)
                        
                        self.after(0, lambda: messagebox.showinfo("Thành công", f"Đã lưu khuôn mặt của {face_name}", parent=self))
                        face_added = True
                        break 
                        
                    except Exception as e:
                        self.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi lưu khuôn mặt: {e}", parent=self))
                        break
            
            elif key == ord('q'):
                self.after(0, lambda: messagebox.showinfo("Thông báo", "Đã hủy thêm khuôn mặt.", parent=self))
                break
        
        cap.release()
        cv2.destroyWindow(add_face_window_name)
        
        if face_added:
            save_reference_faces(reference_faces) # Hàm toàn cục

    def scan_face_thread_start(self):
        if not (self.facenet_model and self.resnet18_model):
            messagebox.showerror("Lỗi", "Mô hình chưa được tải xong hoặc có lỗi.", parent=self)
            return
        threading.Thread(target=self._scan_and_identify_faces, daemon=True).start()

    def _scan_and_identify_faces(self):
        reference_faces = load_reference_faces() # Hàm toàn cục
        
        if not reference_faces:
            self.after(0, lambda: messagebox.showwarning("Thiếu dữ liệu", "Chưa có khuôn mặt tham chiếu! Hãy thêm khuôn mặt trước.", parent=self))
            return
            
        cap1 = cv2.VideoCapture(0)
        if not cap1.isOpened():
            self.after(0, lambda: messagebox.showerror("Lỗi Camera 1", "Không thể mở camera 1.", parent=self))
            return

        cap2 = cv2.VideoCapture(1)
        has_second_camera = cap2.isOpened()
        
        if not has_second_camera:
            print("Không tìm thấy camera thứ hai, sẽ hiển thị kết quả của cả hai mô hình trên cùng một camera.")
        
        facenet_window_name = "FaceNet Model - Nhan \'q\' de thoat"
        resnet_window_name = "ResNet18 Model - Nhan \'q\' de thoat"
        cv2.namedWindow(facenet_window_name)
        cv2.namedWindow(resnet_window_name)

        try:
            while True:
                ret1, frame1 = cap1.read()
                if not ret1:
                    self.after(0, lambda: messagebox.showerror("Lỗi Camera 1", "Không thể đọc frame từ camera 1.", parent=self))
                    break
                
                frame2 = None
                if has_second_camera:
                    ret2, frame2_cap = cap2.read()
                    if not ret2:
                        self.after(0, lambda: messagebox.showerror("Lỗi Camera 2", "Không thể đọc frame từ camera 2.", parent=self))
                        # Có thể muốn break ở đây hoặc chỉ tiếp tục với camera 1
                        # For now, let's allow it to continue with cam1 if cam2 fails mid-loop
                        frame2 = frame1.copy() # Fallback to frame1 if cam2 fails
                    else:
                        frame2 = frame2_cap
                else:
                    frame2 = frame1.copy()
            
                facenet_frame_display = frame1.copy()
                resnet_frame_display = frame2.copy()
                
                faces1 = detect_faces(frame1) # detect on original frame1
                faces2 = detect_faces(frame2) # detect on original frame2
                
                # Xử lý khuôn mặt với FaceNet
                for (x, y, w, h) in faces1:
                    face_img = frame1[y:y+h, x:x+w] # crop from original frame1
                    if face_img.size > 0:
                        try:
                            facenet_emb = get_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), self.facenet_model)
                            best_match_fn, best_sim_fn = None, -1
                            for name, data in reference_faces.items():
                                sim = cosine_similarity(facenet_emb, data['facenet_embedding'])
                                if sim > best_sim_fn: best_sim_fn, best_match_fn = sim, name
                            
                            current_facenet_threshold = self.facenet_threshold_var.get()
                            cv2.rectangle(facenet_frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            text_fn = f"{best_match_fn if best_sim_fn > current_facenet_threshold else 'Unknown'}: {best_sim_fn:.2f}"
                            color_fn = (0, 255, 0) if best_sim_fn > current_facenet_threshold else (0, 0, 255)
                            cv2.putText(facenet_frame_display, text_fn, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_fn, 2)
                        except Exception as e: print(f"FaceNet processing error: {e}")
                
                # Xử lý khuôn mặt với ResNet18
                for (x, y, w, h) in faces2:
                    face_img = frame2[y:y+h, x:x+w] # crop from original frame2
                    if face_img.size > 0:
                        try:
                            resnet_emb = get_embedding(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), self.resnet18_model)
                            best_match_rn, best_sim_rn = None, -1
                            for name, data in reference_faces.items():
                                sim = cosine_similarity(resnet_emb, data['resnet_embedding'])
                                if sim > best_sim_rn: best_sim_rn, best_match_rn = sim, name

                            current_resnet_threshold = self.resnet_threshold_var.get()
                            cv2.rectangle(resnet_frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            text_rn = f"{best_match_rn if best_sim_rn > current_resnet_threshold else 'Unknown'}: {best_sim_rn:.2f}"
                            color_rn = (0, 255, 0) if best_sim_rn > current_resnet_threshold else (0, 0, 255)
                            cv2.putText(resnet_frame_display, text_rn, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rn, 2)
                        except Exception as e: print(f"ResNet processing error: {e}")
                
                cv2.putText(facenet_frame_display, "FaceNet + Triplet Loss", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(facenet_frame_display, "Nhan \'q\' de quay lai", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(resnet_frame_display, "ResNet18 + ArcFace", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(resnet_frame_display, "Nhan \'q\' de quay lai", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
                cv2.imshow(facenet_window_name, facenet_frame_display)
                cv2.imshow(resnet_window_name, resnet_frame_display)
                
                if cv2.waitKey(1) == ord('q'): break
        finally:
            cap1.release()
            if has_second_camera: cap2.release()
            cv2.destroyAllWindows()
            
    def quit_app(self):
        if messagebox.askokcancel("Xác nhận thoát", "Bạn có chắc chắn muốn thoát ứng dụng?", parent=self):
            self.destroy()

if __name__ == "__main__":
    # Thiết lập ban đầu cho thiết bị (device) nên ở đây nếu nó không phụ thuộc vào GUI
    # Tuy nhiên, các print() liên quan đến CUDA có thể gây nhiễu nếu GUI là mục tiêu chính
    # For now, device setup is at global scope as before.
    app = FaceApp()
    app.mainloop()
