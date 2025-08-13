import threading
import cv2
import time
import queue
import os
import shutil
from retinaface import RetinaFace

class FrameProcessor:
    def __init__(self):
        self.processed_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.processing_enabled = False
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
        self.saved_face_count = 0
        
        # Initialize RetinaFace model
        # The model will be automatically downloaded on first use
        print("Initializing RetinaFace model...")
        # Run a dummy detection to trigger model download
        dummy_frame = cv2.imread("/dev/null", cv2.IMREAD_COLOR)
        if dummy_frame is None:
            dummy_frame = cv2.imread("/tmp/nonexistent.jpg", cv2.IMREAD_COLOR)
        print("RetinaFace model initialized.")
        
        # Set up directory for cropped face images
        self.cropped_images_dir = "cropped_images"
        self._setup_cropped_images_dir()

    def _setup_cropped_images_dir(self):
        """Clean up and create the cropped images directory"""
        if os.path.exists(self.cropped_images_dir):
            shutil.rmtree(self.cropped_images_dir)
        os.makedirs(self.cropped_images_dir)

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

            # Process the frame with RetinaFace face detection
            processed = self._detect_faces(frame)

            with self.lock:
                self.processed_frame = processed
                self.frame_count += 1

    def _detect_faces(self, frame):
        """Detect faces using RetinaFace, draw green rectangles around them, and save cropped faces"""
        try:
            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()
            
            # Run face detection
            # Note: RetinaFace expects RGB format, but OpenCV uses BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_frame)
            
            # Draw rectangles around all detected faces and save cropped faces
            face_count = 0
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, face_data["facial_area"])
                    
                    # Draw green rectangle around the face
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_count += 1
                    
                    # Draw facial landmarks if available
                    if "landmarks" in face_data:
                        for landmark_key, landmark in face_data["landmarks"].items():
                            lx, ly = map(int, landmark)
                            cv2.circle(annotated_frame, (lx, ly), 2, (0, 0, 255), -1)
                    
                    # Crop and save the face
                    self._save_cropped_face(frame, x1, y1, x2, y2)
            
            # Add face count text
            cv2.putText(annotated_frame, f"Faces detected: {face_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize to a standard size while maintaining aspect ratio
            target_height = 480
            aspect_ratio = annotated_frame.shape[1] / annotated_frame.shape[0]
            target_width = int(target_height * aspect_ratio)
            
            resized_frame = cv2.resize(annotated_frame, (target_width, target_height))
            
            return resized_frame
        except Exception as e:
            # Handle any errors during face detection
            print(f"Error in face detection: {e}")
            # Return original frame resized to standard size
            return cv2.resize(frame, (640, 480))

    def _save_cropped_face(self, frame, x1, y1, x2, y2):
        """Save a cropped face image to the cropped_images directory"""
        try:
            # Add margin around the face (10% of face size)
            width = x2 - x1
            height = y2 - y1
            margin_x = int(width * 0.1)
            margin_y = int(height * 0.1)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(frame.shape[1], x2 + margin_x)
            y2 = min(frame.shape[0], y2 + margin_y)
            
            # Crop the face region
            cropped_face = frame[y1:y2, x1:x2]
            
            # Save the cropped face
            filename = f"face_{self.saved_face_count:06d}.jpg"
            filepath = os.path.join(self.cropped_images_dir, filename)
            cv2.imwrite(filepath, cropped_face)
            self.saved_face_count += 1
        except Exception as e:
            print(f"Error saving cropped face: {e}")

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
