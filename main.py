import cv2
import time
import queue
import threading
from stream import VideoStream
from frame_processor import FrameProcessor
from ui_controller import UIController
from embedder import embedder  # Import the global embedder instance
import numpy as np
from camera_detector import detect_cameras

# Import and initialize vector database
import vector_db
vector_db.init_index(dim=512, index_path="faiss_index.bin")
print("🔧 FAISS Vector Database initialized")


def main():
    # Create a queue for configuration updates
    config_queue = queue.Queue()

    # Initialize UI controller
    ui_controller = UIController(config_queue)
    ui_controller.start()

    # Initialize video stream with better defaults
    try:
        # Use the first available camera
        available_cameras = detect_cameras()
        camera_source = available_cameras[0] if available_cameras else 0
        stream = VideoStream(src=camera_source, width=640, height=480, fps=60).start()
    except RuntimeError as e:
        print(f"Error starting video stream: {e}")
        ui_controller.stop()
        return

    processor = FrameProcessor().start()

    # Create windows
    cv2.namedWindow("Original Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processed Feed", cv2.WINDOW_NORMAL)

    last_fps_print = time.time()
    processing_enabled = True
    camera_streaming = True
    processing_active = True
    last_config_check = time.time()

    try:
        while True:
            # Check for configuration updates every 100ms
            if time.time() - last_config_check >= 0.1:
                try:
                    config = config_queue.get_nowait()

                    # Handle processing enabled toggle
                    if "processing_enabled" in config:
                        processing_enabled = config["processing_enabled"]
                        processor.toggle_processing(processing_enabled)

                    # Handle camera streaming control
                    if "camera_streaming" in config:
                        camera_streaming = config["camera_streaming"]

                    # Handle processing active control
                    if "processing_active" in config:
                        processing_active = config["processing_active"]
                        # Only update processing if it's not globally disabled
                        if (
                            "processing_enabled" in config
                        ):  # Only toggle if we have the enabled state
                            if processing_enabled:
                                processor.toggle_processing(processing_active)

                    # Handle face detection parameters
                    if "face_margin_ratio" in config:
                        try:
                            processor.set_face_margin_ratio(config["face_margin_ratio"])
                        except Exception as e:
                            print(f"Warning: Could not set face margin ratio: {e}")

                    if "face_rect_thickness" in config:
                        try:
                            processor.set_face_rect_thickness(
                                config["face_rect_thickness"]
                            )
                        except Exception as e:
                            print(f"Warning: Could not set face rect thickness: {e}")

                    if "landmark_radius" in config:
                        try:
                            processor.set_landmark_radius(config["landmark_radius"])
                        except Exception as e:
                            print(f"Warning: Could not set landmark radius: {e}")

                    # Handle processing mode
                    if "processing_mode" in config:
                        try:
                            if config["processing_mode"] == "fullframe":
                                processor.set_send_full_frame(True)
                                # For full frame mode, we want to skip face detection in the embedder
                                embedder.set_expect_aligned_face(False)
                            else:
                                processor.set_send_full_frame(False)
                                # For aligned mode, we skip face detection in the embedder
                                if config["processing_mode"] == "aligned":
                                    embedder.set_expect_aligned_face(True)
                                else:
                                    embedder.set_expect_aligned_face(False)
                        except Exception as e:
                            print(f"Warning: Could not set processing mode: {e}")

                    # Handle preprocessing parameters
                    if 'convert_to_rgb' in config:
                        try:
                            embedder.set_convert_to_rgb(config['convert_to_rgb'])
                        except Exception as e:
                            print(f"Warning: Could not set convert to RGB: {e}")
                        
                    if 'target_width' in config and 'target_height' in config:
                        try:
                            embedder.set_target_size(config['target_width'], config['target_height'])
                        except Exception as e:
                            print(f"Warning: Could not set target size: {e}")
                    
                    # Handle recognition parameters
                    if 'continuous_scanning' in config:
                        try:
                            processor.set_continuous_scanning(config['continuous_scanning'])
                        except Exception as e:
                            print(f"Warning: Could not set continuous scanning: {e}")
                    
                    if 'recognition_threshold' in config:
                        try:
                            processor.set_recognition_threshold(config['recognition_threshold'])
                        except Exception as e:
                            print(f"Warning: Could not set recognition threshold: {e}")

                    # Handle frame capture request for user registration
                    if 'capture_frame' in config and config['capture_frame']:
                        print("📸 Received frame capture request from registration window")
                        print(f"📊 Camera state - streaming: {camera_streaming}, has_new_frame: {stream.has_new_frame()}")
                        if camera_streaming and stream.has_new_frame():
                            print("📥 Attempting to read frame from camera stream")
                            ret, frame = stream.read()
                            if ret:
                                print(f"✅ Frame captured successfully - shape: {frame.shape}")
                                # Send frame to UI controller for user registration
                                print("📤 Sending captured frame to UI controller")
                                ui_controller.set_captured_frame(frame)
                            else:
                                print("❌ Failed to read frame from camera stream")
                        elif not camera_streaming:
                            print("⏸️ Camera is currently paused - attempting to capture frame anyway")
                            # Try to read a frame even if camera is marked as paused
                            # This can happen if the pause is temporary or UI state is out of sync
                            print("📥 Attempting to read frame from camera stream (paused state)")
                            ret, frame = stream.read()
                            if ret:
                                print(f"✅ Frame captured successfully (forced) - shape: {frame.shape}")
                                # Send frame to UI controller for user registration
                                print("📤 Sending captured frame to UI controller")
                                ui_controller.set_captured_frame(frame)
                            else:
                                print("❌ Failed to read frame from camera stream (forced attempt)")
                        else:
                            print("⏳ No new frame available - attempting to wait for one")
                            # Try to wait a bit for a new frame
                            frame_wait_start = time.time()
                            while time.time() - frame_wait_start < 0.5:  # Wait up to 500ms
                                if stream.has_new_frame():
                                    print("📥 New frame detected, attempting to read")
                                    ret, frame = stream.read()
                                    if ret:
                                        print(f"✅ Frame captured successfully (after wait) - shape: {frame.shape}")
                                        # Send frame to UI controller for user registration
                                        print("📤 Sending captured frame to UI controller")
                                        ui_controller.set_captured_frame(frame)
                                        break
                                    else:
                                        print("❌ Failed to read frame from camera stream (after wait)")
                                        break
                                time.sleep(0.01)  # Small delay to prevent busy waiting
                            else:
                                print("⏰ Timeout waiting for new frame - giving up")

                    # If camera settings changed, restart stream
                    if (
                        "camera_source" in config
                        and "width" in config
                        and "height" in config
                        and "fps" in config
                        and (
                            config["camera_source"] != stream.src
                            or config["width"] != stream.width
                            or config["height"] != stream.height
                            or config["fps"] != stream.fps
                        )
                    ):
                        # Stop current stream
                        stream.stop()

                        # Start new stream with updated settings
                        stream = VideoStream(
                            src=int(config["camera_source"]),
                            width=config["width"],
                            height=config["height"],
                            fps=config["fps"],
                        ).start()

                        ui_controller.update_status("Camera settings updated")

                except queue.Empty:
                    pass
                last_config_check = time.time()

            # Only process if there's a new frame and camera streaming is enabled
            if camera_streaming and stream.has_new_frame():
                # Get frame from stream
                ret, frame = stream.read()
                if not ret:
                    print("Failed to get frame from camera")
                    break

                # Display original frame
                cv2.imshow("Original Feed", frame)

                # Submit for processing only if processing is enabled and active
                if processing_enabled and processing_active:
                    processor.process_frame(frame)

                # Get and display processed frame if available
                processed = processor.get_processed_frame()
                if processed is not None and processing_enabled and processing_active:
                    cv2.imshow("Processed Feed", processed)
                elif not processing_active and processing_enabled:
                    # Show blank frame when processing is paused but enabled
                    blank = np.zeros(
                        (frame.shape[0], frame.shape[1], 3), dtype=np.uint8
                    )
                    cv2.putText(
                        blank,
                        "Processing Paused",
                        (blank.shape[1] // 2 - 120, blank.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Processed Feed", blank)
            elif not camera_streaming:
                # Show blank frame when camera streaming is paused
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Camera Paused",
                    (blank.shape[1] // 2 - 120, blank.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Original Feed", blank)

                # Also show blank for processed feed
                blank2 = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank2,
                    "Camera Paused",
                    (blank2.shape[1] // 2 - 120, blank2.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Processed Feed", blank2)

                # Small delay to prevent excessive CPU usage when camera is paused
                time.sleep(0.01)
            else:
                # Small delay to prevent excessive CPU usage when no new frame is available
                time.sleep(0.001)

            # Print FPS every second
            if time.time() - last_fps_print >= 1.0:
                # print(f"Camera FPS: {stream.get_fps():.1f}, Processing FPS: {processor.get_fps():.1f}")
                last_fps_print = time.time()

            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                # Toggle processing
                processing_enabled = not processing_enabled
                processor.toggle_processing(processing_enabled)
                print("Processing:", "Enabled" if processing_enabled else "Disabled")

                if not processing_enabled:
                    # Show blank frame when processing is disabled
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Processing Disabled",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Processed Feed", blank)

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        stream.stop()
        processor.stop()
        ui_controller.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
