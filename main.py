import cv2
import time
from stream import VideoStream
from frame_processor import FrameProcessor
import numpy as np

def main():
    # Initialize video stream with better defaults
    try:
        stream = VideoStream(width=640, height=480, fps=60).start()
    except RuntimeError as e:
        print(f"Error starting video stream: {e}")
        return
        
    processor = FrameProcessor().start()
    
    # Create windows
    cv2.namedWindow('Original Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processed Feed', cv2.WINDOW_NORMAL)
    
    last_fps_print = time.time()
    processing_enabled = True
    
    try:
        while True:
            # Only process if there's a new frame
            if stream.has_new_frame():
                # Get frame from stream
                ret, frame = stream.read()
                if not ret:
                    print("Failed to get frame from camera")
                    break
                    
                # Display original frame
                cv2.imshow('Original Feed', frame)
                
                # Submit for processing only if processing is enabled
                if processing_enabled:
                    processor.process_frame(frame)
                
                # Get and display processed frame if available
                processed = processor.get_processed_frame()
                if processed is not None and processing_enabled:
                    cv2.imshow('Processed Feed', processed)
            else:
                # Small delay to prevent excessive CPU usage when no new frame is available
                time.sleep(0.001)
            
            # Print FPS every second
            if time.time() - last_fps_print >= 1.0:
                print(f"Camera FPS: {stream.get_fps():.1f}, Processing FPS: {processor.get_fps():.1f}")
                last_fps_print = time.time()
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Toggle processing
                processing_enabled = not processing_enabled
                processor.toggle_processing(processing_enabled)
                print("Processing:", "Enabled" if processing_enabled else "Disabled")
                
                if not processing_enabled:
                    # Show blank frame when processing is disabled
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Processing Disabled", (180, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('Processed Feed', blank)
                
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        stream.stop()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
