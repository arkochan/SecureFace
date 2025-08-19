import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import threading
import queue


class FaceEmbedder:
    def __init__(self):
        """Initialize the FaceEmbedder with ArcFace model running on CPU"""
        print("🔧 Initializing FaceEmbedder with CPU...")

        # Initialize with CPU only and specify ArcFace model
        # Use a larger detection size like in the test script to avoid tensor shape issues
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))  # Use 640x640 like test script

        # Preprocessing parameters
        self.convert_to_rgb = True
        self.target_size = (112, 112)  # For recognition model
        self.expect_aligned_face = False  # Whether to expect an already aligned face
        self.skip_detection_for_aligned = True  # Skip detection for pre-aligned faces

        # Queue for handling embeddings in a separate thread
        self.embedding_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Start the embedding thread
        self.embedding_thread = threading.Thread(
            target=self._embedding_worker, daemon=True
        )
        self.embedding_thread.start()

        print("✅ FaceEmbedder initialized successfully")
    
    def _embedding_worker(self):
        """Worker function that runs in a separate thread to process embeddings"""
        print("🔧 Embedding worker thread started")
        while True:
            try:
                # Get cropped face and timestamp from queue
                item = self.embedding_queue.get()
                if item is None:
                    break
                
                cropped_face, timestamp = item
                
                # Process the embedding
                embedding = self._process_embedding(cropped_face)
                
                # Put result and timestamp in result queue
                self.result_queue.put((embedding, timestamp))
                
                # Mark task as done
                self.embedding_queue.task_done()
            except Exception as e:
                print(f"❌ Error in embedding worker: {e}")
                self.result_queue.put((None, None))
    
    def _process_embedding(self, cropped_face):
        """Process a single cropped face to generate embedding using ArcFace"""
        try:
            print(f"📥 Processing cropped face with shape: {cropped_face.shape}")

            # Convert BGR to RGB (InsightFace expects RGB) - configurable
            if self.convert_to_rgb:
                img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                print(f"🔄 Converted to RGB, shape: {img_rgb.shape}")
            else:
                img_rgb = cropped_face
                print(f"⏭️  Skipped RGB conversion, using original image")

            # If we expect an already aligned face and want to skip detection
            if self.expect_aligned_face and self.skip_detection_for_aligned:
                try:
                    # For pre-aligned faces, resize to target size and use recognition model directly
                    resized_img = cv2.resize(img_rgb, self.target_size)
                    print(f"📏 Resized to: {resized_img.shape}")
                    
                    # Call recognition model directly for pre-aligned faces
                    embedding = self.app.models['recognition'].get(resized_img)
                    print(f"✅ Generated ArcFace embedding with shape: {embedding.shape}")
                    return embedding
                except Exception as direct_error:
                    print(f"⚠️ Direct embedding failed: {direct_error}")
                    # Fall back to full pipeline if direct approach fails

            # For normal operation, use the full pipeline with proper detection size
            # The app was prepared with det_size=(640, 640) to avoid tensor issues
            faces = self.app.get(img_rgb)
            print(f"👤 Faces detected by ArcFace: {len(faces)}")

            # Check if we got a face
            if len(faces) > 0:
                # ArcFace embedding
                embedding = faces[0].embedding
                print(f"✅ Generated ArcFace embedding with shape: {embedding.shape}")
                return embedding
            else:
                # Create debug directory if it doesn't exist
                import os
                debug_dir = "debug_images"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                
                # Save the image for debugging if no face detected
                import time

                timestamp = int(time.time() * 1000) % 1000000
                debug_filename = f"{debug_dir}/debug_no_face_{timestamp}.jpg"
                cv2.imwrite(debug_filename, cropped_face)
                print(f"⚠️ No face detected by ArcFace, saved image as {debug_filename}")

                # Also save the RGB version for comparison
                rgb_filename = f"{debug_dir}/debug_no_face_{timestamp}_rgb.jpg"
                if self.convert_to_rgb:
                    cv2.imwrite(rgb_filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(rgb_filename, img_rgb)
                print(f"💾 Saved RGB image as {rgb_filename} for debugging")
                return None
        except Exception as e:
            # Create debug directory if it doesn't exist
            import os
            debug_dir = "debug_images"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Save the image for debugging on error
            import time

            timestamp = int(time.time() * 1000) % 1000000
            debug_filename = f"{debug_dir}/debug_error_{timestamp}.jpg"
            cv2.imwrite(debug_filename, cropped_face)
            print(
                f"❌ Error processing ArcFace embedding, saved image as {debug_filename}: {e}"
            )
            import traceback
            traceback.print_exc()
            return None
    
    def embed(self, cropped_face, timestamp):
        """
        Send a cropped face for embedding processing.
        This method is thread-safe and non-blocking.

        Args:
            cropped_face (numpy.ndarray): Cropped face image from frame
            timestamp (float): The timestamp of the frame
            
        Returns:
            None: Result will be available in the result queue
        """
        # Put the cropped face and timestamp in the queue for processing
        self.embedding_queue.put((cropped_face, timestamp))
    
    def embed_direct(self, image, timestamp):
        """
        Send a full image (not necessarily a cropped face) for direct embedding processing.
        This method is thread-safe and non-blocking.

        Args:
            image (numpy.ndarray): Full image frame
            timestamp (float): The timestamp of the frame

        Returns:
            None: Result will be available in the result queue
        """
        # Put the full image and timestamp in the queue for processing
        self.embedding_queue.put((image, timestamp))

    def set_convert_to_rgb(self, enabled):
        """Enable or disable RGB conversion"""
        self.convert_to_rgb = enabled

    def set_target_size(self, width, height):
        """Set the target size for image resizing"""
        self.target_size = (width, height)

    def set_expect_aligned_face(self, enabled):
        """Enable or disable expecting an already aligned face"""
        self.expect_aligned_face = enabled

    def get_preprocessing_params(self):
        """Get the current preprocessing parameters"""
        return {
            'convert_to_rgb': self.convert_to_rgb,
            'target_size': self.target_size,
            'expect_aligned_face': self.expect_aligned_face
        }

    def get_embedding_result(self, timeout=None):
        """
        Get the latest embedding result from the queue.
        
        Args:
            timeout (float, optional): Timeout in seconds to wait for result
            
        Returns:
            (numpy.ndarray or None, float or None): The face embedding and timestamp, or (None, None) if error/timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def get_all_embedding_results(self, timeout=0.1):
        """
        Get all available embedding results from the queue.
        
        Args:
            timeout (float): Timeout in seconds for each result retrieval
            
        Returns:
            list: List of (face embedding, timestamp) tuples
        """
        results = []
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def stop(self):
        """Stop the embedding worker thread"""
        self.embedding_queue.put(None)  # Signal to stop
        if self.embedding_thread.is_alive():
            self.embedding_thread.join()


# Global instance for easy access
embedder = FaceEmbedder()