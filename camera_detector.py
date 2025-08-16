import cv2

def detect_cameras(max_cameras=10):
    """
    Detect available cameras by trying to open each camera index.
    
    Args:
        max_cameras (int): Maximum number of camera indices to test
        
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to make sure the camera actually works
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
            cap.release()
        except Exception:
            # Camera not available or error occurred
            pass
    
    return available_cameras

if __name__ == "__main__":
    cameras = detect_cameras()
    if cameras:
        print(f"Available cameras: {cameras}")
        for cam in cameras:
            print(f"  Camera {cam}: /dev/video{cam}")
    else:
        print("No cameras found")