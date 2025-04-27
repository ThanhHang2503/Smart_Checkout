import os
import pandas as pd
import random
import shutil
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

# Đường dẫn tới LFW
lfw_root = r"D:\dataset\lfwver2\lfw-deepfunneled"
people_train_file = r"D:\dataset\lfwver2\peopleDevTrain.csv"
people_test_file = r"D:\dataset\lfwver2\peopleDevTest.csv"
matchpairs_train_file = r"D:\dataset\lfwver2\matchpairsDevTrain.csv"
mismatchpairs_train_file = r"D:\dataset\lfwver2\mismatchpairsDevTrain.csv"
matchpairs_test_file = r"D:\dataset\lfwver2\matchpairsDevTest.csv"
mismatchpairs_test_file = r"D:\dataset\lfwver2\mismatchpairsDevTest.csv"

# Thư mục lưu trữ các tập train, validation, test
output_dir = r"D:\dataset\lfwver2"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")

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

# 3. Đọc danh sách danh tính từ file CSV
def load_identities(people_train_file, people_test_file):
    # Kiểm tra sự tồn tại của file và in thông tin chi tiết
    if not os.path.exists(people_train_file):
        print(f"File not found: {people_train_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory {os.path.dirname(people_train_file)}:")
        print(os.listdir(os.path.dirname(people_train_file)))
        raise FileNotFoundError(f"File not found: {people_train_file}")
    
    if not os.path.exists(people_test_file):
        print(f"File not found: {people_test_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in directory {os.path.dirname(people_test_file)}:")
        print(os.listdir(os.path.dirname(people_test_file)))
        raise FileNotFoundError(f"File not found: {people_test_file}")
    
    train_identities_df = pd.read_csv(people_train_file)
    test_identities_df = pd.read_csv(people_test_file)
    
    train_identities = train_identities_df['name'].tolist()
    test_identities = test_identities_df['name'].tolist()
    
    return train_identities, test_identities

# 4. Chia tập dữ liệu
def split_lwf_dataset(root_dir, train_identities, test_identities, train_ratio=0.7):
    train_ids, val_ids = train_test_split(
        train_identities, train_size=train_ratio, random_state=42
    )
    test_ids = test_identities
    
    all_identities = train_ids + val_ids + test_ids
    identity_to_label = {identity: idx for idx, identity in enumerate(all_identities)}
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for identity in all_identities:
        person_dir = os.path.join(root_dir, identity)
        if os.path.isdir(person_dir):
            person_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
            label = identity_to_label[identity]
            
            if identity in train_ids:
                train_paths.extend(person_images)
                train_labels.extend([label] * len(person_images))
            elif identity in val_ids:
                val_paths.extend(person_images)
                val_labels.extend([label] * len(person_images))
            elif identity in test_ids:
                test_paths.extend(person_images)
                test_labels.extend([label] * len(person_images))
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

# 5. Lưu dữ liệu vào thư mục
def save_split_data(train_data, val_data, test_data, train_dir, val_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Lưu tập train
    train_paths, train_labels = train_data
    for img_path, label in zip(train_paths, train_labels):
        identity = os.path.basename(os.path.dirname(img_path))
        dest_dir = os.path.join(train_dir, identity)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)
    
    # Lưu tập validation
    val_paths, val_labels = val_data
    for img_path, label in zip(val_paths, val_labels):
        identity = os.path.basename(os.path.dirname(img_path))
        dest_dir = os.path.join(val_dir, identity)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)
    
    # Lưu tập test
    test_paths, test_labels = test_data
    for img_path, label in zip(test_paths, test_labels):
        identity = os.path.basename(os.path.dirname(img_path))
        dest_dir = os.path.join(test_dir, identity)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)
    
    print(f"Saved train data to: {train_dir}")
    print(f"Saved validation data to: {val_dir}")
    print(f"Saved test data to: {test_dir}")

# 6. Tạo DataLoader
def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    train_dataset = FaceDataset(train_paths, train_labels, transform=transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=transform)
    test_dataset = FaceDataset(test_paths, test_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# 7. Kiểm tra overlap giữa các tập
def check_overlap(train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    print(f"Overlap between train and validation: {len(train_set.intersection(val_set))} identities")
    print(f"Overlap between train and test: {len(train_set.intersection(test_set))} identities")
    print(f"Overlap between validation and test: {len(val_set.intersection(test_set))} identities")

# 8. Đọc danh tính và chia dữ liệu
try:
    train_identities, test_identities = load_identities(people_train_file, people_test_file)
except FileNotFoundError as e:
    print(e)
    print("Please check the file paths and ensure the files exist.")
    exit(1)

(train_data, val_data, test_data) = split_lwf_dataset(lfw_root, train_identities, test_identities)

# 9. Lưu dữ liệu vào thư mục
save_split_data(train_data, val_data, test_data, train_dir, val_dir, test_dir)

# 10. Tạo DataLoader
train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

# 11. In thông tin phân chia
def print_split_info(data, name):
    paths, labels = data
    print(f"{name} set: {len(paths)} images, {len(set(labels))} identities")

print_split_info(train_data, "Train")
print_split_info(val_data, "Validation")
print_split_info(test_data, "Test")

# 12. Kiểm tra overlap
train_ids = set([person for person in train_identities if any(img_path.startswith(os.path.join(lfw_root, person)) for img_path, _ in zip(train_data[0], train_data[1]))])
val_ids = set([person for person in train_identities if any(img_path.startswith(os.path.join(lfw_root, person)) for img_path, _ in zip(val_data[0], val_data[1]))])
test_ids = set([person for person in test_identities if any(img_path.startswith(os.path.join(lfw_root, person)) for img_path, _ in zip(test_data[0], test_data[1]))])
check_overlap(train_ids, val_ids, test_ids)