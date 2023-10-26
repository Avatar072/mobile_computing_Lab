import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from kivy.app import App
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget

from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.togglebutton import ToggleButton
from kivy.utils import get_color_from_hex
from PIL import Image as PILImage
from MNIST_test import Net

class DrawCanvasWidget(Widget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # 默认划线的颜色
        # self.canvas.add(Color(rgb=[0,0,0]))
        self.change_color(get_color_from_hex('#19caad'))  # 修改默认划线线颜色
        self.line_width = 10

    def on_touch_down(self,touch):
        if Widget.on_touch_down(self,touch):
            return
        with self.canvas:
            touch.ud['current_line'] = Line(points=(touch.x,touch.y),width=self.line_width)

    def on_touch_move(self,touch):
        if 'current_line' in touch.ud:
            touch.ud['current_line'].points += (touch.x,touch.y)

    def change_color(self,new_color):
        self.last_color = new_color  # 在清除画板时使用
        self.canvas.add(Color(*new_color))

    def change_line_width(self,line_width="Normal"):
        self.line_width = {"Thin":1,"Normal":2,"Thick":4}[line_width]


    def clean_canvas(self):
        saved = self.children[:]  # 保留root控件上的子控件（button)
        self.clear_widgets()  # 清除所有控件间
        self.canvas.clear()   # 清除canvas
        for widget in saved:  # 将清除的子控件再画上
            self.add_widget(widget)

        self.change_color(self.last_color)
    
    def save_canvas(self, filename):
        # 保存画布之前，将元件从画布上移除
        saved = self.children[:]
        self.clear_widgets()

        # 导出画布为图像
        texture = self.export_as_image()
        texture.save(filename)

        # 保存完成后，重新添加元件
        for widget in saved:
            self.add_widget(widget)



class PaintApp(App):
    def build(self):
        self.draw_canvas_widget = DrawCanvasWidget()

       
        # 将模型设置为评估模式
        self.model = Net()
        self.model.load_state_dict(torch.load("mnist_cnn.pt"))
        self.model.eval()

        return self.draw_canvas_widget  # 返回root控件

    def save_canvas(self):
        filename = "canvas_image.png"
        self.draw_canvas_widget.save_canvas(filename)

           # 读取存储的图像
        image = cv2.imread("canvas_image.png", cv2.IMREAD_GRAYSCALE)
        # 获取图像的形状
        height, width = image.shape
        # 如果是彩色图像，还可以获取通道数
        # height, width, channels = image.shape
        print(f"图像高度：{height}")
        print(f"图像宽度：{width}")
        # 调整图像大小
        resized_image = cv2.resize(image, (28, 28)).astype(np.uint8)  # 将图像转换为无符号8位整数

        # 保存调整大小后的图像
        cv2.imwrite("resized_image.png", resized_image)
        image = cv2.resize(image, (28, 28)).astype(np.float32) / 255.0
        image = image[None,:]  # 调整形状为 [1, 1, 28, 28]
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # 进行推理
        with torch.no_grad():
            output = self.model(image_tensor)
            # output = model(image_tensor)  # 使用导入的模型
        
        predicted_class = output.argmax().item()
        print("Predicted class:", predicted_class)
        
if __name__ == "__main__":
    # # 加载保存的模型
    model = Net()
    # # model.eval()
    PaintApp().run()