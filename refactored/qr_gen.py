from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SwapTransition
from kivy.properties import StringProperty, ListProperty
from kivy.uix.recycleview import RecycleView
from kivy.core.clipboard import Clipboard
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import qrcode
from qrcode.image.pil import PilImage
import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
from pathlib import Path
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
import logging
from functools import wraps
from typing import Any, Optional
import time
from contextlib import contextmanager
from kivy.animation import Animation
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

DOCUMENTS_PATH = str(Path.home() / "Documents")
PROJECT_ROOT = Path(r"C:\Users\Mark Wayne Cleofe\OneDrive - Camarines Sur Polytechnic Colleges\Desktop\Defensive Programming\python-kivy-qr-reader-and-generator\refactored")
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename='qr_app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Update window background color
Window.clearcolor = get_color_from_hex('#1565C0')  # Material Blue 800

def show_copy_popup():
    popup = Popup(
        title='Copied!',
        content=Label(text='Text copied to clipboard'),
        size_hint=(None, None),
        size=(300, 150)
    )
    popup.open()

def show_error_popup(message: str) -> None:
    """Display error message in popup"""
    popup = Popup(
        title='Error',
        content=Label(text=str(message)),
        size_hint=(None, None),
        size=(300, 150)
    )
    popup.open()
    logging.error(f"Error popup shown: {message}")

def show_success_popup(message: str) -> None:
    """Display success message in popup"""
    popup = Popup(
        title='Success',
        content=Label(text=str(message)),
        size_hint=(None, None),
        size=(300, 150)
    )
    popup.open()
    logging.info(f"Success popup shown: {message}")

class InputValidator:
    @staticmethod
    def validate_qr_data(data: str) -> bool:
        return bool(data and len(data) <= 2953)

    @staticmethod
    def sanitize_input(data: str) -> str:
        return ''.join(c for c in data if c.isprintable())

def rate_limit(max_calls: int, time_frame: float):
    def decorator(func):
        last_calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            last_calls.append(now)
            last_calls[:] = [t for t in last_calls if t > now - time_frame]
            if len(last_calls) > max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_frame} seconds")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def resource_handler():
    try:
        yield
    except Exception as e:
        logging.error(f"Resource error: {str(e)}")
        raise
    finally:
        logging.info("Resources cleaned up")

class MenuScreen(Screen):
    pass


class ReadQR(Screen):
    default_path = StringProperty(str(OUTPUT_DIR))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load(self, path, filename):
        try:
            if not filename:
                App.get_running_app().txtReadData = "Please select a file"
                return
                
            file_path = os.path.join(path, filename[0])
            if not os.path.exists(file_path):
                App.get_running_app().txtReadData = "File not found"
                return
                
            loaded_code = cv2.imread(file_path)
            if loaded_code is None:
                App.get_running_app().txtReadData = "Invalid image file"
                return
                
            decoded_objects = decode(loaded_code)
            
            if not decoded_objects:
                App.get_running_app().txtReadData = "No QR code found in image"
                return
                
            data = decoded_objects[0].data
            App.get_running_app().txtReadData = data.decode("UTF-8")
            App.get_running_app().add_to_history(data.decode("UTF-8"), "Read QR")
            
        except Exception as e:
            App.get_running_app().txtReadData = f"Error: {str(e)}"


class DataScreen(Screen):
    def copy_text(self):
        app = App.get_running_app()
        if app.txtReadData:
            Clipboard.copy(app.txtReadData)
            show_copy_popup()
            
    def go_to_menu(self):
        self.manager.current = 'menu'


def safe_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters"""
    if not filename:
        return "untitled"
    
    filename = os.path.basename(filename)
    invalid_chars = '<>:"/\\|?*'
    return ''.join('_' if c in invalid_chars else c for c in filename)

class GenerateQR(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.version = 'Auto'
        self.error_level = 'M'
        self.validator = InputValidator()
    
    def copy_to_clipboard(self):
        try:
            data = self.ids.qr_data.text.strip()
            if not data:
                show_error_popup("No text to copy")
                logging.warning("Attempted to copy empty text")
                return
                
            Clipboard.copy(data)
            show_success_popup("Text copied to clipboard")
            logging.info(f"Text copied to clipboard: {data[:50]}...")
            
        except Exception as e:
            logging.error(f"Error copying to clipboard: {str(e)}")
            show_error_popup("Failed to copy text")

    @rate_limit(max_calls=5, time_frame=60)
    def genQR(self):
        try:
            data = self.ids.qr_data.text.strip()
            
            if not data:
                show_error_popup("Please enter data to generate QR code")
                return
                
            if not self.validator.validate_qr_data(data):
                show_error_popup("Invalid QR data")
                return
            
            sanitized_data = self.validator.sanitize_input(data)
            
            with resource_handler():
                SUPPORTED_VERSIONS = {
                    'Auto': None,
                    'Version 1': 1,
                    'Version 2': 2, 
                    'Version 3': 3
                }
                
                if self.version not in SUPPORTED_VERSIONS:
                    raise ValueError("Invalid QR version")
                    
                qr = qrcode.QRCode(
                    version=SUPPORTED_VERSIONS[self.version],
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(sanitized_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = safe_filename(f"QR_{timestamp}.png")
                save_path = OUTPUT_DIR / filename
                
                qr.make_image().save(str(save_path))
                logging.info(f"QR code generated: {filename}")
                App.get_running_app().add_to_history(sanitized_data, "Generated QR")
                show_success_popup("QR Code generated successfully!")
                
        except Exception as e:
            show_error_popup(f"Error generating QR code: {str(e)}")
            logging.error(f"QR generation error: {str(e)}")


class CameraQR(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        
    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)
        
    def on_leave(self):
        if self.capture:
            self.capture.release()
            Clock.unschedule(self.update)
            
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # QR Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray)
            
            if decoded_objects:
                data = decoded_objects[0].data.decode("UTF-8")
                App.get_running_app().txtReadData = data
                App.get_running_app().add_to_history(data, "Camera Scan")
                self.manager.current = 'datascreen'
                return False
            
            # Update preview
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.camera_preview.texture = texture


class HistoryRecycleView(RecycleView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        
class HistoryScreen(Screen):
    def on_enter(self):
        self.ids.history_list.data = App.get_running_app().history
        
    def clear_history(self):
        App.get_running_app().history.clear()
        self.ids.history_list.data = []


class KivyQRApp(App):
    txtReadData = StringProperty('')
    history = ListProperty([])
    
    def add_to_history(self, data, action_type):
        self.history.append({
            'text': f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {action_type}: {data[:30]}...",
            'type': action_type,
            'data': data
        })

    def build(self):
        self.sm = ScreenManager(transition=SwapTransition())
        screens = [
            MenuScreen(name='menu'),
            GenerateQR(name='generate'),
            ReadQR(name='read'),
            DataScreen(name='datascreen'),
            CameraQR(name='camera'),  # Add CameraQR screen
            HistoryScreen(name='history')
        ]
        for screen in screens:
            self.sm.add_widget(screen)
        return self.sm

    def switch_screen(self, screen_name, duration=0.3):
        anim = Animation(opacity=0, duration=duration/2) + Animation(opacity=1, duration=duration/2)
        anim.start(self.sm.current_screen)
        self.sm.current = screen_name

    def btnClose(self):
        sys.exit()

    def reloadFiles(self):
        try:
            self.sm.remove_widget(self.read)
            self.read = ReadQR(name='read')
            self.sm.add_widget(self.read)
        except Exception as e:
            print(f"Error reloading files: {str(e)}")


if __name__ == '__main__':
    KivyQRApp().run()

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from qr_gen import InputValidator

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.validator = InputValidator()
    
    def test_input_validation(self):
        test_cases = [
            ("", False),
            ("A" * 3000, False), 
            ("Valid Input", True),
            ("Test Data", True)
        ]
        
        for input_data, expected in test_cases:
            with self.subTest(input=input_data):
                result = self.validator.validate_qr_data(input_data)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
