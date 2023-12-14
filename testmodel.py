import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from MNIST_test import Net

# 1. 加载和预处理图像
# image = cv2.imread("canvas_image.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread('canvas_image_test.png')
# resized_image = cv2.resize(image, (28, 28))

# 2. 数据格式转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 使用与训练数据相同的均值和标准差
])

# input_tensor = transform(resized_image).unsqueeze(0)
input_tensor = transform(image).unsqueeze(1)

# 3. 模型推理
model = Net()  # 创建模型
model.load_state_dict(torch.load("mnist_cnn.pt"))  # 加载预训练权重
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    output = model(input_tensor)  # 进行推理

print(output.argmax().item())
# 4. 结果解释
predicted_class = output.argmax().item()
print("Predicted class:", predicted_class)