# -*- coding: utf-8 -*-
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.app import App
from kivy.graphics.texture import Texture

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image as PilImage

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

# 加載預先訓練的模型權重
model.load_state_dict(torch.load('./model/simple_cnn_model.pth'))
model.eval()

class ImageUploadLayout(BoxLayout):
    file_chooser = FileChooserListView()

    def __init__(self, **kwargs):
        super(ImageUploadLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # 指定本地端 PNG 檔案的路徑
        local_image_path = './background/dog_background.png'

        # 使用 Image 類別顯示背景
        self.background = Image(source=local_image_path, allow_stretch=True, keep_ratio=True)
        self.add_widget(self.background)

        # 返回按鈕
        return_button = Button(text='Back', size_hint=(None, None), size=(50, 50), pos_hint={'right': 1, 'top': 1})
        return_button.bind(on_release=self.return_to_main_page)
        self.add_widget(return_button)

        # 上傳按鈕
        upload_button = Button(text='UpLoad', size_hint=(None, None), size=(50, 50), pos_hint={'right': 1, 'top': 0.9})
        upload_button.bind(on_release=self.select_image)
        self.add_widget(upload_button)

        # 辨識按鈕
        recognition_button = Button(text='identification', size_hint=(None, None), size=(100, 50), pos_hint={'right': 1, 'top': 0.8})
        recognition_button.bind(on_release=self.on_recognize_button_click)
        self.add_widget(recognition_button)


    def select_image(self, instance):
        # 開啟檔案選擇器
        self.file_chooser = FileChooserListView()
        self.file_chooser.bind(on_submit=self.load_image)
        popup = Popup(title='Select an Image', content=self.file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_image(self, instance, selection, touch):
        if len(selection) > 0:
            image_path = selection[0]

            # 檢查所選檔案是否為圖片
            if self.is_image_file(image_path):
                # 彈窗詢問是否確定選擇這張圖片
                self.show_confirmation("Confirm Selection", "Are you sure to select this image?", image_path)
            else:
                # 如果不是圖片，彈出警告訊息
                self.show_warning("Please Choose a Picture!!!")

    def upload_image(self, selected, image_path):
        if selected:
            # 背景圖變成上傳的圖片
            self.background.source = image_path

            # 新增：對上傳的圖片進行模型評估
            # 這邊方便上傳圖片時測試辨識功能用
            # self.evaluate_image(image_path)

            print("Image uploaded!")
        else:
            print("Cancelled image upload.")

    def evaluate_image(self, image_path):
        # 新增：對上傳的圖片進行模型評估的程式碼
        # 虛構一個輸入圖片和對應的標籤（根據您的實際需求調整）
        input_image = PilImage.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_data = transform(input_image).unsqueeze(0)

        # 透過模型進行預測
        with torch.no_grad():
            output = model(input_data)

        # 將模型的輸出轉換為標籤
        predicted_label = output.argmax(dim=1).item()

        # 打印預測結果
        print("模型預測結果:", "貓" if predicted_label == 0 else "狗")

        # 顯示預測結果
        self.show_prediction_result(predicted_label)

    # 在 ImageUploadLayout 类中添加一个新的方法，用于处理辨識按鈕的点击事件
    def on_recognize_button_click(self, instance):
        # 获取当前背景图路径
        image_path = self.background.source
        if image_path != './background/dog_background.png':
            # 调用 evaluate_image 方法进行模型评估
            self.evaluate_image(image_path)
        else:
            # 提示用户需要先上传图片
            self.show_warning("Please upload an image first!")

    # 新增方法來顯示預測結果
    def show_prediction_result(self, predicted_label):
        if predicted_label == 0:
            result_message = "predicted result: Cat"
        else:
            result_message = "predicted result: Dog"

        # 彈窗顯示預測結果
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=result_message))

        button = Button(text='OK', size_hint=(1, None), height='48dp')
        content.add_widget(button)

        popup = Popup(title='Prediction Result', content=content, size_hint=(None, None), size=(300, 200))
        button.bind(on_release=popup.dismiss)  # 關閉彈窗的動作
        popup.open()

    def return_to_main_page(self, instance):
        if self.background.source == './background/dog_background.png':
            # 已在首頁，彈窗提示
            self.show_warning("Already on the Main Page!")
        else:
            # 將背景圖恢復成預設的圖片
            self.background.source = './background/dog_background.png'
            print("Returned to Main Page!")

    def is_image_file(self, file_path):
        # 檢查檔案擴展名是否為圖片格式，這裡只是簡單示範，實際應用中可能需要更複雜的檢查邏輯
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)

    def show_warning(self, message):
        # 彈出警告彈窗，帶有確定按鈕
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))

        button = Button(text='OK', size_hint=(1, None), height='48dp')
        content.add_widget(button)

        popup = Popup(title='Warning', content=content, size_hint=(None, None), size=(300, 200))
        button.bind(on_release=popup.dismiss)  # 關閉彈窗的動作
        popup.open()

    def show_confirmation(self, title, message, image_path):
        # 彈出確認彈窗，帶有Yes和No按鈕
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))

        yes_button = Button(text='Yes', size_hint=(1, None), height='48dp')
        no_button = Button(text='No', size_hint=(1, None), height='48dp')

        yes_button.bind(on_release=lambda instance: self.upload_image(True, image_path))
        yes_button.bind(on_release=lambda instance: popup.dismiss())
        no_button.bind(on_release=lambda instance: popup.dismiss())

        content.add_widget(yes_button)
        content.add_widget(no_button)

        popup = ConfirmationPopup(title=title, content=content, size_hint=(None, None), size=(300, 200))
        popup.image_path = image_path  # 儲存圖片路徑供後續使用
        popup.open()

    def on_size(self, instance, value):
        # 在這裡調整背景圖的大小
        self.background.size = self.size

class ConfirmationPopup(Popup):
    image_path = ''  # 新增屬性用於儲存圖片路徑

class ImageUploadApp(App):
    def build(self):
        return ImageUploadLayout()

if __name__ == '__main__':
    ImageUploadApp().run()
