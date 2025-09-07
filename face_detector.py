import cv2
import numpy as np
from abc import ABC, abstractmethod
import os

class FaceDetector(ABC):
    """Abstract base class for face detection implementations"""
    
    @abstractmethod
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected faces with bounding boxes and landmarks
        """
        pass
    
    @abstractmethod
    def draw_faces(self, frame, faces, rect_color=(0, 255, 0), rect_thickness=2, 
                   landmark_radius=2, landmark_color=(0, 0, 255), landmark_thickness=-1):
        """
        Draw detected faces on a frame
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            faces (list): List of detected faces
            rect_color (tuple): Rectangle color (B, G, R)
            rect_thickness (int): Rectangle thickness
            landmark_radius (int): Landmark circle radius
            landmark_color (tuple): Landmark color (B, G, R)
            landmark_thickness (int): Landmark circle thickness
            
        Returns:
            numpy.ndarray: Frame with drawn faces
        """
        pass


class RetinaFaceDetector(FaceDetector):
    """RetinaFace implementation of FaceDetector"""
    
    def __init__(self):
        try:
            from retinaface import RetinaFace
            self.retinaface = RetinaFace
        except ImportError:
            raise ImportError("RetinaFace not installed. Please install retinaface-package")
    
    def detect_faces(self, frame):
        """
        Detect faces using RetinaFace
        
        Returns:
            list: List of face dictionaries with 'bbox' and 'landmarks' keys
        """
        # RetinaFace expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_result = self.retinaface.detect_faces(rgb_frame)
        
        faces = []
        if isinstance(faces_result, dict):
            for face_key, face_data in faces_result.items():
                faces.append({
                    'bbox': face_data['facial_area'],
                    'landmarks': face_data.get('landmarks', {})
                })
        
        return faces
    
    def draw_faces(self, frame, faces, rect_color=(0, 255, 0), rect_thickness=2,
                   landmark_radius=2, landmark_color=(0, 0, 255), landmark_thickness=-1):
        """Draw faces detected by RetinaFace"""
        annotated_frame = frame.copy()
        
        for face in faces:
            # Draw rectangle around face
            x1, y1, x2, y2 = map(int, face['bbox'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color, rect_thickness)
            
            # Draw landmarks
            for landmark_key, landmark in face['landmarks'].items():
                lx, ly = map(int, landmark)
                cv2.circle(annotated_frame, (lx, ly), landmark_radius, 
                          landmark_color, landmark_thickness)
        
        return annotated_frame


class InsightFaceDetector(FaceDetector):
    """InsightFace implementation of FaceDetector"""
    
    def __init__(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except ImportError:
            raise ImportError("InsightFace not installed. Please install insightface")
    
    def detect_faces(self, frame):
        """
        Detect faces using InsightFace
        
        Returns:
            list: List of face objects with bbox and landmark attributes
        """
        # InsightFace works directly with BGR frames
        faces = self.app.get(frame)
        return faces
    
    def draw_faces(self, frame, faces, rect_color=(0, 255, 0), rect_thickness=2,
                   landmark_radius=2, landmark_color=(0, 0, 255), landmark_thickness=-1):
        """Draw faces detected by InsightFace"""
        annotated_frame = frame.copy()
        
        for face in faces:
            # Draw rectangle around face
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color, rect_thickness)
            
            # Draw landmarks (using 2D 106-point landmarks if available)
            if hasattr(face, 'landmark_2d_106'):
                for landmark in face.landmark_2d_106:
                    lx, ly = map(int, landmark)
                    cv2.circle(annotated_frame, (lx, ly), landmark_radius,
                              landmark_color, landmark_thickness)
            # Fallback to 3D 68-point landmarks
            elif hasattr(face, 'landmark_3d_68'):
                for landmark in face.landmark_3d_68[:, :2]:  # Only x, y coordinates
                    lx, ly = map(int, landmark)
                    cv2.circle(annotated_frame, (lx, ly), landmark_radius,
                              landmark_color, landmark_thickness)
        
        return annotated_frame


class SCRFDdetector(FaceDetector):
    """SCRFD (Scale-aware Receptive Field Discovery) implementation of FaceDetector"""
    
    def __init__(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            # SCRFD is the detector used in buffalo_l model by default
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except ImportError:
            raise ImportError("InsightFace not installed. Please install insightface")
    
    def detect_faces(self, frame):
        """
        Detect faces using SCRFD (through InsightFace app)
        
        Returns:
            list: List of face objects with bbox and landmark attributes
        """
        # SCRFD works directly with BGR frames
        faces = self.app.get(frame)
        return faces
    
    def draw_faces(self, frame, faces, rect_color=(0, 255, 0), rect_thickness=2,
                   landmark_radius=2, landmark_color=(0, 0, 255), landmark_thickness=-1):
        """Draw faces detected by SCRFD"""
        annotated_frame = frame.copy()
        
        for face in faces:
            # Draw rectangle around face
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rect_color, rect_thickness)
            
            # Draw landmarks (using 2D 106-point landmarks if available)
            if hasattr(face, 'landmark_2d_106'):
                for landmark in face.landmark_2d_106:
                    lx, ly = map(int, landmark)
                    cv2.circle(annotated_frame, (lx, ly), landmark_radius,
                              landmark_color, landmark_thickness)
            # Fallback to 3D 68-point landmarks
            elif hasattr(face, 'landmark_3d_68'):
                for landmark in face.landmark_3d_68[:, :2]:  # Only x, y coordinates
                    lx, ly = map(int, landmark)
                    cv2.circle(annotated_frame, (lx, ly), landmark_radius,
                              landmark_color, landmark_thickness)
        
        return annotated_frame


class FaceDetectorFactory:
    """Factory for creating face detector instances"""
    
    @staticmethod
    def create_detector(detector_type="retinaface"):
        """
        Create a face detector instance
        
        Args:
            detector_type (str): Type of detector ('retinaface', 'insightface', or 'scrfd')
            
        Returns:
            FaceDetector: Instance of the requested detector
        """
        if detector_type.lower() == 'retinaface':
            return RetinaFaceDetector()
        elif detector_type.lower() == 'insightface':
            return InsightFaceDetector()
        elif detector_type.lower() == 'scrfd':
            return SCRFDdetector()
        else:
            # Default to RetinaFace if unknown type
            print(f"Unknown detector type: {detector_type}. Using RetinaFace as default.")
            return RetinaFaceDetector()