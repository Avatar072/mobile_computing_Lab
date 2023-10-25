

'''
xujing
2020-06-30

基于kivymd和pytorch的二次元风格转换app

'''

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.properties import ObjectProperty

from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivymd.uix.picker import MDDatePicker
from kivymd.uix.menu import MDDropdownMenu, RightContent

from kivymd.toast import toast
from kivymd.uix.bottomsheet import MDGridBottomSheet

from kivymd.uix.filemanager import MDFileManager
from kivy.graphics.texture import Texture
from kivy.uix.button import Button  # 添加导入
from kivy.lang import Builder


import requests
import base64
import json
import cv2
import numpy as np
import time
import os
import logging
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView

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
        
        self.file_chooser = FileChooserListView()
        self.file_chooser.bind(on_submit=self.load_image)
    
    def select_image(self, instance):
        popup = Popup(title='Select an Image', content=self.file_chooser, size_hint=(0.9, 0.9))
        popup.open()
    
    def load_image(self, instance, selection, touch):
        if len(selection) > 0:
            image_path = selection[0]
            self.image_preview.source = image_path
    
    def go_back(self, instance):
        # 添加返回按钮的处理逻辑
        self.image_preview.source = ''  # 清空图像预览

class ImageUploadApp(App):
    def build(self):
        return ImageUploadLayout()

if __name__ == '__main__':
    ImageUploadApp().run()