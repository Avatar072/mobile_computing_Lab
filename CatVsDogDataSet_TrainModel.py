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

# 检查是否有可用的 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 训练数据集，最多载入1000张猫和狗的图片
train_cat_dataset = CatDogDataset(root_dir='./PetImages/Train/Cat/', transform=transform, max_images=5000)
train_dog_dataset = CatDogDataset(root_dir='./PetImages/Train/Dog/', transform=transform, max_images=5000)

# 合并猫和狗的训练数据集
train_dataset = torch.utils.data.ConcatDataset([train_cat_dataset, train_dog_dataset])

# 训练数据集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

# 测试数据集，最多载入500张猫和狗的图片
test_cat_dataset = CatDogDataset(root_dir='./PetImages/Test/Cat/', transform=transform, max_images=1000)
test_dog_dataset = CatDogDataset(root_dir='./PetImages/Test/Dog/', transform=transform, max_images=1000)

# 合并猫和狗的测试数据集
test_dataset = torch.utils.data.ConcatDataset([test_cat_dataset, test_dog_dataset])

# 测试数据集的 DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 定义简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 2)  # 根据你的图片大小调整这里的数值

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
# 稍复杂的 CNN 模型，具有四个卷积层和三个全连接层
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

# 实例化模型、损失函数和优化器
# model = SimpleCNN()
model = ComplexCNN()    
model.to(device)  # 将模型移动到 GPU 上
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(true_labels, predictions)

# 显示混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 计算准确率
accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 显示分类报告
print('Classification Report:')
print(classification_report(true_labels, predictions, target_names=['Cat', 'Dog']))

# 保存模型的状态字典和其他信息
torch.save(model.state_dict(), './model/simple_cnn_model.pth')
