import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1  # Sử dụng InceptionResnetV1 từ facenet-pytorch

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

# 4. Định nghĩa mô hình FaceNet (sử dụng InceptionResnetV1 từ facenet-pytorch)
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2')  # Sử dụng trọng số pretrained trên VGGFace2
        self.model.logits = nn.Linear(1792, embedding_size)  # Thay đổi lớp cuối cùng
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # Chuẩn hóa L2
        return x

# 5. Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# 6. Huấn luyện mô hình
def train_facenet(model, train_loader, val_loader, epochs=20):
    model = model.to(device)
    criterion = TripletLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Tạo triplet (anchor, positive, negative)
            triplets = []
            for i in range(len(labels)):
                anchor_img = images[i].unsqueeze(0)
                anchor_label = labels[i]
                
                # Tìm positive (cùng danh tính)
                positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                positive_indices = positive_indices[positive_indices != i]
                if len(positive_indices) == 0:
                    continue
                positive_idx = np.random.choice(positive_indices.cpu().numpy())
                positive_img = images[positive_idx].unsqueeze(0)
                
                # Tìm negative (khác danh tính)
                negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                if len(negative_indices) == 0:
                    continue
                negative_idx = np.random.choice(negative_indices.cpu().numpy())
                negative_img = images[negative_idx].unsqueeze(0)
                
                triplets.append((anchor_img, positive_img, negative_img))
            
            if not triplets:
                continue
            
            anchors, positives, negatives = zip(*triplets)
            anchors = torch.cat(anchors).to(device)
            positives = torch.cat(positives).to(device)
            negatives = torch.cat(negatives).to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    
    # Lưu mô hình
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/facenet.pth")

# 7. Khởi tạo và huấn luyện
model = FaceNet(embedding_size=128)
train_facenet(model, train_loader, val_loader, epochs=20)