import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io

# Thiết lập mã hóa UTF-8 cho console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Đường dẫn tệp
matchpairs_train_file = r"D:\dataset\lfwver2\matchpairsDevTrain.csv"
mismatchpairs_train_file = r"D:\dataset\lfwver2\mismatchpairsDevTrain.csv"

# Tải dữ liệu
try:
    match_df = pd.read_csv(matchpairs_train_file, encoding='utf-8')
    mismatch_df = pd.read_csv(mismatchpairs_train_file, encoding='utf-8')
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy tệp. Vui lòng kiểm tra đường dẫn.\n{e}")
    exit()
except UnicodeDecodeError:
    print("Lỗi mã hóa: Thử đọc tệp với mã hóa 'latin1' hoặc kiểm tra định dạng tệp.")
    exit()

# 1. Thông tin cơ bản
print("=== Dữ liệu Cặp Khớp (Train) ===")
print("Kích thước:", match_df.shape)
print("\nCác cột:", match_df.columns.tolist())
print("\n5 hàng đầu tiên:")
print(match_df.head())
print("\nThông tin:")
match_df.info()
print("\nGiá trị thiếu:")
print(match_df.isnull().sum())

print("\n=== Dữ liệu Cặp Không Khớp (Train) ===")
print("Kích thước:", mismatch_df.shape)
print("\nCác cột:", mismatch_df.columns.tolist())
print("\n5 hàng đầu tiên:")
print(mismatch_df.head())
print("\nThông tin:")
mismatch_df.info()
print("\nGiá trị thiếu:")
print(mismatch_df.isnull().sum())

# 2. Thống kê mô tả
print("\n=== Thống kê Mô tả ===")
print("Cặp Khớp (Train):")
print(match_df.describe(include='all'))
print("\nCặp Không Khớp (Train):")
print(mismatch_df.describe(include='all'))

# 3. Phân tích giá trị duy nhất
print("\n=== Giá trị Duy nhất trong Cặp Khớp (Train) ===")
for col in match_df.columns:
    print(f"{col}: {match_df[col].nunique()} giá trị duy nhất")
    if match_df[col].dtype == 'object':
        print(f"5 giá trị phổ biến nhất trong {col}:\n{match_df[col].value_counts().head()}")

print("\n=== Giá trị Duy nhất trong Cặp Không Khớp (Train) ===")
for col in mismatch_df.columns:
    print(f"{col}: {mismatch_df[col].nunique()} giá trị duy nhất")
    if mismatch_df[col].dtype == 'object':
        print(f"5 giá trị phổ biến nhất trong {col}:\n{mismatch_df[col].value_counts().head()}")

# 4. Trực quan hóa
plt.figure(figsize=(12, 6))

# Vẽ tần suất tên trong cặp khớp (giả sử có cột 'name')
if 'name' in match_df.columns:
    plt.subplot(1, 2, 1)
    match_df['name'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Tên trong Cặp Khớp (Train)')
    plt.xlabel('Tên')
    plt.ylabel('Tần suất')
    plt.xticks(rotation=45)

# Vẽ tần suất tên trong cặp không khớp (giả sử có cột 'name1')
if 'name1' in mismatch_df.columns:
    plt.subplot(1, 2, 2)
    mismatch_df['name1'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Tên trong Cặp Không Khớp (Train, Name1)')
    plt.xlabel('Tên')
    plt.ylabel('Tần suất')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 5. Phân tích phân phối cặp
print("\n=== Phân phối Cặp (Train) ===")
print(f"Số cặp khớp: {len(match_df)}")
print(f"Số cặp không khớp: {len(mismatch_df)}")

# Trực quan hóa phân phối cặp
plt.figure(figsize=(6, 4))
sns.barplot(x=['Cặp Khớp', 'Cặp Không Khớp'], y=[len(match_df), len(mismatch_df)])
plt.title('Phân phối Cặp Khớp vs. Không Khớp (Train)')
plt.ylabel('Số lượng Cặp')
plt.show()