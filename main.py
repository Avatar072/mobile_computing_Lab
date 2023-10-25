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

class DrawCanvasWidget(Widget):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # 默认划线的颜色
        # self.canvas.add(Color(rgb=[0,0,0]))
        self.change_color(get_color_from_hex('#19caad'))  # 修改默认划线线颜色
        self.line_width = 2

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
        texture = self.export_as_image()
        texture.save(filename)





class PaintApp(App):
    def build(self):
        # 加载保存的模型
        # model = torch.load("mnist_cnn.pt")
        # # 将模型设置为评估模式
        # model.eval()
        
        self.draw_canvas_widget = DrawCanvasWidget()

        return self.draw_canvas_widget  # 返回root控件

    def save_canvas(self):
        filename = "canvas_image.png"
        self.draw_canvas_widget.save_canvas(filename)
        
if __name__ == "__main__":
    PaintApp().run()