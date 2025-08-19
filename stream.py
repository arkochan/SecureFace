import cv2
import threading
import time
import numpy as np

def list_cameras(max_cams=5):
    available = []
    for i in range(max_cams):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available.append(f"Camera {i}")
                cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
    if not available:
        print("No cameras found. Please check your camera connections.")
    return available

class VideoStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        self.new_frame = False

    def start(self):
        self.stopped = False
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.src}")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read initial frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame")
        
        with self.lock:
            self.frame = (frame, time.time())
            self.new_frame = True
            
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = (frame, time.time())
                self.new_frame = True
                self.frame_count += 1

    def read(self):
        with self.lock:
            if self.frame is not None:
                self.new_frame = False
                return True, self.frame[0], self.frame[1]
            return False, None, None
            
    def has_new_frame(self):
        with self.lock:
            return self.new_frame

    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0
