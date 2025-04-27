import os
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới thư mục LFW
lfw_root = "D:/code/FaceNet/Smart_Checkout/LFW/lfw-funneled/"  # Đường dẫn tới thư mục lfw-funneled
pairs_train_file = "D:/code/FaceNet/Smart_Checkout/LFW/pairsDevTrain.txt"  # Đường dẫn tới pairsDevTrain.txt
pairs_test_file = "D:/code/FaceNet/Smart_Checkout/LFW/pairsDevTest.txt"  # Đường dẫn tới pairsDevTest.txt

# 1. Lớp Dataset tùy chỉnh
class LWFDataset(Dataset):
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
    transforms.Resize((160, 160)),  # Kích thước cho FaceNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. Chia tập dữ liệu theo danh tính
def split_lwf_dataset(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    identities = []
    image_paths = []
    labels = []
    
    for idx, person in enumerate(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if os.path.isdir(person_dir):
            identities.append(person)
            for img_name in os.listdir(person_dir):
                image_paths.append(os.path.join(person_dir, img_name))
                labels.append(idx)
    
    # Chia danh tính thành train, val, test
    train_ids, temp_ids = train_test_split(
        identities, train_size=train_ratio, random_state=42
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_ratio_adjusted, random_state=42
    )
    
    # Tạo danh sách ảnh và nhãn cho từng tập
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for idx, person in enumerate(identities):
        person_dir = os.path.join(root_dir, person)
        person_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
        person_label = idx
        
        if person in train_ids:
            train_paths.extend(person_images)
            train_labels.extend([person_label] * len(person_images))
        elif person in val_ids:
            val_paths.extend(person_images)
            val_labels.extend([person_label] * len(person_images))
        elif person in test_ids:
            test_paths.extend(person_images)
            test_labels.extend([person_label] * len(person_images))
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# 4. Tạo DataLoader
def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    train_dataset = LWFDataset(train_paths, train_labels, transform=transform)
    val_dataset = LWFDataset(val_paths, val_labels, transform=transform)
    test_dataset = LWFDataset(test_paths, test_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# 5. Đọc file pairs (pairsDevTrain.txt hoặc pairsDevTest.txt)
def load_pairs(pairs_file, lfw_root):
    pairs = []
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
        num_folds, num_pairs = map(int, lines[0].split())
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:  # Cặp cùng danh tính
                person, idx1, idx2 = parts
                img1_path = os.path.join(lfw_root, person, f"{person}_{int(idx1):04d}.jpg")
                img2_path = os.path.join(lfw_root, person, f"{person}_{int(idx2):04d}.jpg")
                label = 1  # Cùng danh tính
            elif len(parts) == 4:  # Cặp khác danh tính
                person1, idx1, person2, idx2 = parts
                img1_path = os.path.join(lfw_root, person1, f"{person1}_{int(idx1):04d}.jpg")
                img2_path = os.path.join(lfw_root, person2, f"{person2}_{int(idx2):04d}.jpg")
                label = 0  # Khác danh tính
            pairs.append((img1_path, img2_path, label))
    return pairs

# 6. Thực hiện chia dữ liệu
(train_data, val_data, test_data) = split_lwf_dataset(lfw_root)
train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

# 7. In thông tin phân chia
def print_split_info(data, name):
    paths, labels = data
    print(f"{name} set: {len(paths)} images, {len(set(labels))} identities")

print_split_info(train_data, "Train")
print_split_info(val_data, "Validation")
print_split_info(test_data, "Test")

# 8. Tải các cặp từ pairsDevTrain.txt và pairsDevTest.txt
train_pairs = load_pairs(pairs_train_file, lfw_root)
test_pairs = load_pairs(pairs_test_file, lfw_root)
print(f"Number of training pairs: {len(train_pairs)}")
print(f"Number of test pairs: {len(test_pairs)}")