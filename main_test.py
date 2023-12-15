# -*- coding: utf-8 -*-
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.app import App

class ImageUploadLayout(BoxLayout):
    file_chooser = FileChooserListView()

    def __init__(self, **kwargs):
        super(ImageUploadLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'

        self.main_layout = BoxLayout(orientation='vertical')
        self.add_widget(self.main_layout)

        self.image_preview = Image(source='', size_hint=(1, 0.8))
        self.image_preview.bind(on_touch_down=self.upload_image)
        self.main_layout.add_widget(self.image_preview)

        upload_button = Button(text='Upload Image', size_hint_y=None, height='48dp')
        upload_button.bind(on_release=self.select_image)
        self.main_layout.add_widget(upload_button)

        return_button = Button(text='Return to Main Page', size_hint_y=None, height='48dp')
        return_button.bind(on_release=self.return_to_main_page)
        self.main_layout.add_widget(return_button)

    def select_image(self, instance):
        self.file_chooser = FileChooserListView()
        self.file_chooser.bind(on_submit=self.load_image)
        popup = Popup(title='Select an Image', content=self.file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_image(self, instance, selection, touch):
        if len(selection) > 0:
            image_path = selection[0]

            # 檢查所選檔案是否為圖片
            if self.is_image_file(image_path):
                self.image_preview.source = image_path
                self.image_preview.id = 'image_preview'
                self.upload_image(None, None)
            else:
                # 如果不是圖片，彈出警告訊息
                self.show_warning("Please Choose a Picture!!!")

    def upload_image(self, instance, touch):
        if self.image_preview.source:
            print("Image uploaded!")

    def return_to_main_page(self, instance):
        self.image_preview.source = ''
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


class ImageUploadApp(App):
    def build(self):
        return ImageUploadLayout()

if __name__ == '__main__':
    ImageUploadApp().run()
