# -*- coding: utf-8 -*-
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.app import App

class ImageUploadLayout(BoxLayout):
    file_chooser = FileChooserListView()

    def __init__(self, **kwargs):
        super(ImageUploadLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        upload_button = Button(text='Upload Image', size_hint_y=None, height='48dp')
        upload_button.bind(on_release=self.select_image)
        self.add_widget(upload_button)

        self.image_preview = Image(source='')
        self.image_preview.bind(on_touch_down=self.on_image_click)
        self.add_widget(self.image_preview)

    def select_image(self, instance):
        self.file_chooser = FileChooserListView()
        self.file_chooser.bind(on_submit=self.load_image)
        popup = Popup(title='Select an Image', content=self.file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_image(self, instance, selection, touch):
        if len(selection) > 0:
            image_path = selection[0]
            self.image_preview.source = image_path
            self.image_preview.id = 'image_preview'  # 設置 id 屬性

    def on_image_click(self, instance, touch):
        if self.image_preview.collide_point(*touch.pos):
            print("Image clicked!")  # 在這裡添加點擊圖片後的操作

class ImageUploadApp(App):
    def build(self):
        return ImageUploadLayout()

if __name__ == '__main__':
    ImageUploadApp().run()
