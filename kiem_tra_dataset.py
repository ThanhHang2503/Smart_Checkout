import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Đường dẫn tới các thư mục
train_dir = r"D:\dataset\Dataset\Dataset\train"
val_dir = r"D:\dataset\Dataset\Dataset\validation"
test_dir = r"D:\dataset\Dataset\Dataset\test"
raw_dir = r"D:\dataset\Dataset\Dataset\Raw"

# Hàm kiểm tra số lượng danh tính và ảnh
def check_dataset(directory):
    identities = []
    image_paths = []
    
    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if os.path.isdir(person_dir):
            identities.append(person)
            for img_name in os.listdir(person_dir):
                image_paths.append(os.path.join(person_dir, img_name))
    
    return identities, image_paths

# Kiểm tra từng tập
train_identities, train_images = check_dataset(train_dir)
val_identities, val_images = check_dataset(val_dir)
test_identities, test_images = check_dataset(test_dir)

# In thông tin
print(f"Train set: {len(train_images)} images, {len(train_identities)} identities")
print(f"Validation set: {len(val_images)} images, {len(val_identities)} identities")
print(f"Test set: {len(test_images)} images, {len(test_identities)} identities")

# Kiểm tra rò rỉ dữ liệu
train_val_overlap = set(train_identities).intersection(set(val_identities))
train_test_overlap = set(train_identities).intersection(set(test_identities))
val_test_overlap = set(val_identities).intersection(set(test_identities))

print(f"Overlap between train and validation: {len(train_val_overlap)} identities")
print(f"Overlap between train and test: {len(train_test_overlap)} identities")
print(f"Overlap between validation and test: {len(val_test_overlap)} identities")