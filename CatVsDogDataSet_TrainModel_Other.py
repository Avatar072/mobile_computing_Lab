import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)[:max_images] if max_images else os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except PIL.UnidentifiedImageError:
            # 如果遇到无法识别的图片，跳过当前样本，继续取下一个样本
            print(f"Skipped unreadable image: {img_name}")
            return self.__getitem__((idx + 1) % len(self))  # 取下一个样本

        if self.transform:
            image = self.transform(image)

        # 将 'Cat' 文件夹的图片标签设置为 0，'Dog' 文件夹的图片标签设置为 1
        label = 0 if 'Cat' in img_name else 1  # 假设 'Cat' 文件夹表示猫，'Dog' 文件夹表示狗

        return image, label
    
# 定義簡單的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 2)  # 根據你的圖片大小調整這裡的數值

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 載入模型
# model = SimpleCNN()
model = ComplexCNN()    
model.load_state_dict(torch.load('./model/simple_cnn_model.pth'))
model.eval()

# 載入測試數據集
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# 测试数据集，最多载入500张猫和狗的图片
test_cat_dataset = CatDogDataset(root_dir='./PetImages/Test/Cat/', transform=transform, max_images=1000)
test_dog_dataset = CatDogDataset(root_dir='./PetImages/Test/Dog/', transform=transform, max_images=1000)

# 合并猫和狗的测试数据集
test_dataset = torch.utils.data.ConcatDataset([test_cat_dataset, test_dog_dataset])

# 测试数据集的 DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 初始化變數
correct = 0
total = 0

# 進行測試
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 計算 accuracy
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
