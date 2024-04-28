import os
import time

import cv2
import numpy as np
import pyautogui

# from PIL import Image
from screeninfo import get_monitors

from utils import get_top_left_corner_of_active_app


class Monitor:
    def __init__(self, name, width, height, x, y):
        self.name = name
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Monitor(name={self.name}, width={self.width}, height={self.height}, x={self.x}, y={self.y})"

    @property
    def region(self):
        return (self.x, self.y, self.width, self.height)


def get_monitor_info():
    monitors = get_monitors()
    return [Monitor(m.name, m.width, m.height, m.x, m.y) for m in monitors]


def get_active_monitor(app_name, default_monitor=0):
    window_pos = get_top_left_corner_of_active_app(app_name)
    if window_pos == "Window position not available":
        return get_monitor_info()[default_monitor]

    x, y = map(int, window_pos.split(", "))
    for monitor in get_monitor_info():
        if monitor.x <= x < monitor.x + monitor.width and monitor.y <= y < monitor.y + monitor.height:
            return monitor
    raise ValueError(f"Could not find monitor for window position {window_pos}")


# def get_virtual_screen_size():
#     width = height = 0
#     for m in get_monitors():
#         width = max(width, m.x + m.width)
#         height = max(height, m.y + m.height)
#     return width, height


# Folder
folder = "coach/frames"

# Create the frames folder if it doesn't exist
frames_dir = os.path.join(os.getcwd(), folder)
os.makedirs(frames_dir, exist_ok=True)


def screenshot(monitor):
    start = time.time()

    # Take screenshot of other monitor
    screenshot = pyautogui.screenshot(region=monitor.region)

    # Convert the screenshot to a numpy array
    frame = np.array(screenshot)

    # Convert RGB to BGR format for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize the image
    # TODO: look into this
    max_size = 3000
    ratio = max_size / max(frame.shape[1], frame.shape[0])
    new_size = tuple([int(x * ratio) for x in frame.shape[1::-1]])
    resized_img = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{frames_dir}/screenshot_{timestamp}.jpg"

    # Save the frame as an image file
    cv2.imwrite(path, resized_img)
    end = time.time()

    print(f"\n📸 Took screenshot ({end - start:.2f}s)")
    return path
