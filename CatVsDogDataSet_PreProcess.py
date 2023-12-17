import os
import shutil
import random

# 定義資料集路徑
cat_path = './PetImages/Cat/'
dog_path = './PetImages/Dog/'

#從PetImages裡的Cat資料夾中刪除666.jpg、5673.jpg、Thumbs.db
#Dog的部分為11702.jpg、Thumbs.db。
catlist = ['666.jpg', '5673.jpg', 'Thumbs.db']
doglist = ['11702.jpg', 'Thumbs.db']
for i in catlist:
    file_path = os.path.join(cat_path, i)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Removed: {file_path}')
    else:
        print(f'File not found: {file_path}')

for i in doglist:
    file_path = os.path.join(dog_path, i)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'Removed: {file_path}')
    else:
        print(f'File not found: {file_path}')

# 定義新的資料集路徑
train_cat_path = './PetImages/Train/Cat/'
train_dog_path = './PetImages/Train/Dog/'
test_cat_path = './PetImages/Test/Cat/'
test_dog_path = './PetImages/Test/Dog/'

# 確保新的資料集路徑存在
os.makedirs(train_cat_path, exist_ok=True)
os.makedirs(train_dog_path, exist_ok=True)
os.makedirs(test_cat_path, exist_ok=True)
os.makedirs(test_dog_path, exist_ok=True)

# 取得貓和狗的檔案列表
cat_files = os.listdir(cat_path)
dog_files = os.listdir(dog_path)

# 訓練集比例
train_ratio = 0.8

# 複製貓圖片到訓練集和測試集
for cat_file in cat_files:
    if random.random() < train_ratio:
        shutil.copy(os.path.join(cat_path, cat_file), os.path.join(train_cat_path, cat_file))
    else:
        shutil.copy(os.path.join(cat_path, cat_file), os.path.join(test_cat_path, cat_file))

# 複製狗圖片到訓練集和測試集
for dog_file in dog_files:
    if random.random() < train_ratio:
        shutil.copy(os.path.join(dog_path, dog_file), os.path.join(train_dog_path, dog_file))
    else:
        shutil.copy(os.path.join(dog_path, dog_file), os.path.join(test_dog_path, dog_file))
