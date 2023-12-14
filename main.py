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

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup

from kivy.app import App

class ImageUploadLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(ImageUploadLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        
        # 添加返回按钮
        self.back_button = Button(text='Back', size_hint_y=None, height='48dp')
        self.back_button.bind(on_release=self.go_back)
        self.add_widget(self.back_button)
        
        self.upload_button = Button(text='Upload Image', size_hint_y=None, height='48dp')
        self.upload_button.bind(on_release=self.select_image)
        self.add_widget(self.upload_button)
        
        self.image_preview = Image(source='', size_hint=(None, None), size=(300, 300))
        self.add_widget(self.image_preview)  # 将 image_preview 添加到布局中
        
        # 将 FileChooser 添加到一个单独的布局中
        file_chooser_layout = BoxLayout()
        self.file_chooser = FileChooserListView()
        file_chooser_layout.add_widget(self.file_chooser)
        
        self.popup = Popup(title='Select an Image', content=file_chooser_layout, size_hint=(0.9, 0.9))
        self.file_chooser.bind(on_submit=self.load_image)
    
    def select_image(self, instance):
        self.popup.open()
    
    def load_image(self, instance, selection, touch):
        if len(selection) > 0:
            image_path = selection[0]
            self.image_preview.source = image_path
            self.popup.dismiss()
    
    def go_back(self, instance):
        self.image_preview.source = ''  # 清空图像预览

class ImageUploadApp(App):
    def build(self):
        return ImageUploadLayout()

if __name__ == '__main__':
    ImageUploadApp().run()