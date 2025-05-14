import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from facenet_pytorch import InceptionResnetV1

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới dữ liệu và file cặp ảnh
lfw_root = r"D:\dataset\lfwver2\lfw-deepfunneled"
matchpairs_test_file = r"D:\dataset\lfwver2\matchpairsDevTest.csv"
mismatchpairs_test_file = r"D:\dataset\lfwver2\mismatchpairsDevTest.csv"
model_dir = r"models"
facenet_model_path = os.path.join(model_dir, "facenet.pth")
resnet18_model_path = os.path.join(model_dir, "resnet18_arcface.pth")

# Kiểm tra sự tồn tại của file mô hình
if not os.path.exists(facenet_model_path):
    raise FileNotFoundError(f"Model file not found: {facenet_model_path}")
if not os.path.exists(resnet18_model_path):
    raise FileNotFoundError(f"Model file not found: {resnet18_model_path}")

# 1. Lớp Dataset tùy chỉnh (không dùng trực tiếp trong evaluate, nhưng giữ lại cho tính đầy đủ)
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# 2. Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. Định nghĩa mô hình FaceNet
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model.logits = nn.Linear(1792, embedding_size)
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# 4. Định nghĩa mô hình ResNet18+ArcFace
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

# 5. Đọc các cặp ảnh từ file CSV
def load_pairs(matchpairs_file, mismatchpairs_file, lfw_root):
    pairs = []
    
    # Đọc cặp cùng danh tính (positive pairs)
    match_df = pd.read_csv(matchpairs_file)
    print("Columns in matchpairsDevTest.csv:", match_df.columns.tolist())
    
    required_match_columns = ['name', 'imagenum1', 'imagenum2']
    if not all(col in match_df.columns for col in required_match_columns):
        raise ValueError(f"matchpairsDevTest.csv must contain columns: {required_match_columns}")
    
    for _, row in match_df.iterrows():
        person = row['name']
        idx1 = row['imagenum1']
        idx2 = row['imagenum2']
        img1_path = os.path.join(lfw_root, person, f"{person}_{int(idx1):04d}.jpg")
        img2_path = os.path.join(lfw_root, person, f"{person}_{int(idx2):04d}.jpg")
        label = 1  # Cùng danh tính
        pairs.append((img1_path, img2_path, label))
    
    # Đọc cặp khác danh tính (negative pairs)
    mismatch_df = pd.read_csv(mismatchpairs_file)
    print("Columns in mismatchpairsDevTest.csv:", mismatch_df.columns.tolist())
    
    required_mismatch_columns = ['name', 'imagenum1', 'name.1', 'imagenum2']
    if not all(col in mismatch_df.columns for col in required_mismatch_columns):
        raise ValueError(f"mismatchpairsDevTest.csv must contain columns: {required_mismatch_columns}")
    
    for _, row in mismatch_df.iterrows():
        person1 = row['name']
        idx1 = row['imagenum1']
        person2 = row['name.1']
        idx2 = row['imagenum2']
        img1_path = os.path.join(lfw_root, person1, f"{person1}_{int(idx1):04d}.jpg")
        img2_path = os.path.join(lfw_root, person2, f"{person2}_{int(idx2):04d}.jpg")
        label = 0  # Khác danh tính
        pairs.append((img1_path, img2_path, label))
    
    return pairs

# 6. Đánh giá mô hình
def evaluate_model(model, pairs, threshold=0.5):
    model.eval()
    model = model.to(device)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for img1_path, img2_path, label in pairs:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"Warning: Could not load images: {img1_path}, {img2_path}")
                continue
            
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            img1 = transform(img1).unsqueeze(0).to(device)
            img2 = transform(img2).unsqueeze(0).to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Tính khoảng cách Euclidean
            distance = (emb1 - emb2).pow(2).sum(1).sqrt().item()
            pred = 1 if distance < threshold else 0  # Nhỏ hơn threshold -> cùng danh tính
            
            predictions.append(pred)
            true_labels.append(label)
            
            # Lưu điểm số cho ROC (1 - distance: càng nhỏ càng giống)
            predictions.append(1 - distance)
            true_labels.append(label)
    
    # Tính accuracy
    accuracy = accuracy_score(true_labels[::2], predictions[::2])
    
    # Tính ROC-AUC
    fpr, tpr, _ = roc_curve(true_labels[::2], predictions[1::2])
    roc_auc = auc(fpr, tpr)
    
    return accuracy, fpr, tpr, roc_auc

# 7. Tải và đánh giá cả hai mô hình
try:
    test_pairs = load_pairs(matchpairs_test_file, mismatchpairs_test_file, lfw_root)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error loading pairs: {e}")

# Đánh giá FaceNet
facenet_model = FaceNet(embedding_size=128)
facenet_model.load_state_dict(torch.load(facenet_model_path, map_location=device))
facenet_accuracy, facenet_fpr, facenet_tpr, facenet_roc_auc = evaluate_model(facenet_model, test_pairs)

# Đánh giá ResNet18+ArcFace
num_classes = 2826  # Số danh tính trong tập Train (đã sửa từ 2834)
resnet18_model = ResNet18ArcFace(num_classes=num_classes, embedding_size=128)
resnet18_model.load_state_dict(torch.load(resnet18_model_path, map_location=device))
resnet18_accuracy, resnet18_fpr, resnet18_tpr, resnet18_roc_auc = evaluate_model(resnet18_model, test_pairs)

# 8. In kết quả
print(f"FaceNet Accuracy: {facenet_accuracy:.4f}, ROC-AUC: {facenet_roc_auc:.4f}")
print(f"ResNet18+ArcFace Accuracy: {resnet18_accuracy:.4f}, ROC-AUC: {resnet18_roc_auc:.4f}")

# 9. Vẽ biểu đồ ROC
plt.figure(figsize=(8, 6))
plt.plot(facenet_fpr, facenet_tpr, label=f"FaceNet (AUC = {facenet_roc_auc:.2f})")
plt.plot(resnet18_fpr, resnet18_tpr, label=f"ResNet18+ArcFace (AUC = {resnet18_roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
os.makedirs("results", exist_ok=True)
plt.savefig("results/roc_comparison.png")
plt.show()