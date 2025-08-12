import threading
import cv2
import time
import queue

class FrameProcessor:
    def __init__(self):
        self.processed_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.processing_enabled = False
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame

    def start(self):
        self.stopped = False
        self.processing_enabled = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        return self

    def stop(self):
        self.stopped = True
        self.processing_enabled = False

    def toggle_processing(self, enabled):
        self.processing_enabled = enabled
        if not enabled:
            # Clear the queue when processing is disabled
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

    def _process_loop(self):
        while not self.stopped:
            if not self.processing_enabled:
                time.sleep(0.001)
                continue

            try:
                # Wait for a frame with timeout
                frame = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Process the frame (grayscale conversion)
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            with self.lock:
                self.processed_frame = processed
                self.frame_count += 1

    def process_frame(self, frame):
        """Update the current frame to be processed"""
        if not self.processing_enabled:
            return
            
        # Remove any existing frame in queue to avoid backlog
        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
            
        # Add the new frame
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_processed_frame(self):
        """Get the latest processed frame"""
        with self.lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0
