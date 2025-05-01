import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới dữ liệu
train_dir = r"D:\dataset\lfwver2\train"
val_dir = r"D:\dataset\lfwver2\validation"

# 1. Lớp Dataset tùy chỉnh
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

# 3. Tạo dataset và dataloader
def create_dataset(data_dir):
    image_paths = []
    labels = []
    identity_to_label = {}
    label_idx = 0
    
    for identity in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, identity)
        if os.path.isdir(person_dir):
            identity_to_label[identity] = label_idx
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image_paths.append(img_path)
                labels.append(label_idx)
            label_idx += 1
    
    return FaceDataset(image_paths, labels, transform=transform)

train_dataset = create_dataset(train_dir)
val_dataset = create_dataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
        self.arcface = ArcFaceLoss(num_classes, embedding_size)
    
    def forward(self, x, labels=None):
        embeddings = self.model(x)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return embeddings, logits
        return embeddings

# 5. Huấn luyện mô hình
def train_resnet18(model, train_loader, val_loader, epochs=20):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            embeddings, logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    
    # Lưu mô hình
    torch.save(model.state_dict(), "models/resnet18_arcface.pth")

# 6. Khởi tạo và huấn luyện
num_classes = len(set(train_dataset.labels))
model = ResNet18ArcFace(num_classes=num_classes, embedding_size=128)
train_resnet18(model, train_loader, val_loader, epochs=20)