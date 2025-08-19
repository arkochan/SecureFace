import threading
import cv2
import time
import queue
import os
import shutil
from retinaface import RetinaFace
from embedder import embedder  # Import the global embedder instance
import numpy as np

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
        
        # Face detection and cropping parameters
        self.face_rect_color = (0, 255, 0)  # Green rectangle
        self.face_rect_thickness = 2
        self.face_margin_ratio = 0.1  # 10% margin around face
        self.landmark_radius = 2
        self.landmark_color = (0, 0, 255)  # Red landmarks
        self.landmark_thickness = -1  # Filled circle
        
        # Processing parameters
        self.send_full_frame = False  # Whether to send full frame instead of cropped faces
        self.continuous_scanning = False  # Enable continuous face scanning
        self.recognition_threshold = 1.0  # Distance threshold for recognition (lower = stricter)
        self.last_recognition_time = 0
        self.recognition_cooldown = 1.0  # Minimum seconds between recognitions of the same user
        
        # Initialize RetinaFace model
        # The model will be automatically downloaded on first use
        print("Initializing RetinaFace model...")
        # Run a dummy detection to trigger model download
        dummy_frame = cv2.imread("/dev/null", cv2.IMREAD_COLOR)
        if dummy_frame is None:
            dummy_frame = cv2.imread("/tmp/nonexistent.jpg", cv2.IMREAD_COLOR)
        print("RetinaFace model initialized.")

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
                frame, timestamp = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            # Process the frame with RetinaFace face detection
            processed = self._detect_faces(frame, timestamp)

            with self.lock:
                self.processed_frame = processed
                self.frame_count += 1

    def _detect_faces(self, frame, timestamp):
        """Detect faces using RetinaFace, draw green rectangles around them, and process cropped faces with embedder"""
        try:
            # If send_full_frame is enabled, send the entire frame to the embedder
            if self.send_full_frame:
                embedder.embed_direct(frame, timestamp)
                # Create a copy of the frame to draw on
                annotated_frame = frame.copy()
                # Add text indicating full frame mode
                cv2.putText(annotated_frame, "Full Frame Mode", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, self.face_rect_color, 2)
                
                # Resize to a standard size while maintaining aspect ratio
                target_height = 480
                aspect_ratio = annotated_frame.shape[1] / annotated_frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                
                resized_frame = cv2.resize(annotated_frame, (target_width, target_height))
                return resized_frame
            
            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()
            
            # Run face detection
            # Note: RetinaFace expects RGB format, but OpenCV uses BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_frame)
            
            # Draw rectangles around all detected faces and process cropped faces
            face_count = 0
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, face_data["facial_area"])
                    
                    # Draw rectangle around the face with configurable parameters
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                 self.face_rect_color, self.face_rect_thickness)
                    face_count += 1
                    
                    # Draw facial landmarks if available with configurable parameters
                    if "landmarks" in face_data:
                        for landmark_key, landmark in face_data["landmarks"].items():
                            lx, ly = map(int, landmark)
                            cv2.circle(annotated_frame, (lx, ly), self.landmark_radius, 
                                      self.landmark_color, self.landmark_thickness)
                    
                    # Crop and process the face with embedder
                    self._process_cropped_face(frame, x1, y1, x2, y2, timestamp)
            
            # Add face count text
            cv2.putText(annotated_frame, f"Faces detected: {face_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.face_rect_color, 2)
            
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

    def _process_cropped_face(self, frame, x1, y1, x2, y2, timestamp):
        """Process a cropped face image with the embedder"""
        try:
            # Add margin around the face using configurable parameter
            width = x2 - x1
            height = y2 - y1
            margin_x = int(width * self.face_margin_ratio)
            margin_y = int(height * self.face_margin_ratio)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(frame.shape[1], x2 + margin_x)
            y2 = min(frame.shape[0], y2 + margin_y)
            
            # Crop the face region
            cropped_face = frame[y1:y2, x1:x2]
            
            # Check if cropped face is valid
            if cropped_face.size == 0:
                print("❌ Error: Cropped face is empty")
                return
                
            # Send the cropped face to the embedder for processing
            embedder.embed(cropped_face, timestamp)
            self.saved_face_count += 1
            
            # Get the embedding result (non-blocking)
            embedding, timestamp = embedder.get_embedding_result(timeout=0.01)
            if embedding is not None:
                print(f"✅ Generated embedding with shape: {embedding.shape}")
                
                # If continuous scanning is enabled, perform face recognition
                if self.continuous_scanning:
                    self._recognize_face(embedding, timestamp)
                    
        except Exception as e:
            print(f"Error processing cropped face: {e}")

    def _recognize_face(self, embedding, timestamp):
        """Recognize a face by comparing its embedding against the vector database"""
        try:
            # Import vector_db here to avoid circular imports
            import vector_db
            
            # Normalize the embedding
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Search for similar faces in the vector database
            results = vector_db.search_embeddings(normalized_embedding, k=5)
            
            if results:
                # Find the best match below the threshold
                best_match = None
                for faiss_id, distance, user_id, created_at in results:
                    if distance < self.recognition_threshold:
                        if best_match is None or distance < best_match[1]:
                            best_match = (user_id, distance, faiss_id)
                
                if best_match:
                    user_id, distance, faiss_id = best_match
                    current_time = time.time()
                    
                    # Implement cooldown to avoid spamming the same recognition
                    if current_time - self.last_recognition_time > self.recognition_cooldown:
                        latency = (time.time() - timestamp) * 1000
                        print(f"🎯 Face recognized! User ID: {user_id}, Distance: {distance:.4f}, FAISS ID: {faiss_id}, Latency: {latency:.2f} ms")
                        self.last_recognition_time = current_time
                    else:
                        print(f"🔄 Face recognized but in cooldown period - User ID: {user_id}, Distance: {distance:.4f}")
                else:
                    print("🔍 Face detected but no matches found below threshold")
            else:
                print("🔍 Face detected but vector database returned no results")
                
        except Exception as e:
            print(f"Error during face recognition: {e}")
            import traceback
            traceback.print_exc()

    def set_face_rect_color(self, color):
        """Set the color of the face detection rectangle"""
        self.face_rect_color = color

    def set_face_rect_thickness(self, thickness):
        """Set the thickness of the face detection rectangle"""
        self.face_rect_thickness = thickness

    def set_face_margin_ratio(self, ratio):
        """Set the margin ratio around the detected face for cropping"""
        self.face_margin_ratio = ratio

    def set_landmark_radius(self, radius):
        """Set the radius of the facial landmark circles"""
        self.landmark_radius = radius

    def set_landmark_color(self, color):
        """Set the color of the facial landmark circles"""
        self.landmark_color = color

    def set_send_full_frame(self, enabled):
        """Set whether to send full frame instead of cropped faces"""
        self.send_full_frame = enabled

    def set_continuous_scanning(self, enabled):
        """Set whether to enable continuous face scanning"""
        self.continuous_scanning = enabled

    def set_recognition_threshold(self, threshold):
        """Set the recognition threshold (lower = stricter)"""
        self.recognition_threshold = threshold

    def get_current_params(self):
        """Get the current face detection and cropping parameters"""
        return {
            'face_rect_color': self.face_rect_color,
            'face_rect_thickness': self.face_rect_thickness,
            'face_margin_ratio': self.face_margin_ratio,
            'landmark_radius': self.landmark_radius,
            'landmark_color': self.landmark_color,
            'send_full_frame': self.send_full_frame,
            'continuous_scanning': self.continuous_scanning,
            'recognition_threshold': self.recognition_threshold
        }

    def reset_params_to_default(self):
        """Reset all face detection and cropping parameters to their default values"""
        self.face_rect_color = (0, 255, 0)  # Green rectangle
        self.face_rect_thickness = 2
        self.face_margin_ratio = 0.1  # 10% margin around face
        self.landmark_radius = 2
        self.landmark_color = (0, 0, 255)  # Red landmarks
        self.landmark_thickness = -1  # Filled circle
        self.continuous_scanning = False
        self.recognition_threshold = 1.0

    def process_frame(self, frame, timestamp):
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
            self.frame_queue.put_nowait((frame, timestamp))
        except queue.Full:
            pass

    def get_processed_frame(self):
        """Get the latest processed frame"""
        with self.lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0
