import tkinter as tk
from tkinter import ttk
import threading
import queue
import cv2

# Import camera detection function
from camera_detector import detect_cameras


class UIController:
    def __init__(self, config_queue):
        self.config_queue = config_queue
        self.root = None
        self.ui_thread = None
        self.running = False

        # UI variables will be initialized in the UI thread
        self.camera_source = None
        self.width = None
        self.height = None
        self.fps = None
        self.processing_enabled = None
        self.status_label = None

        # Control state variables
        self.camera_streaming = True
        self.processing_active = True

        # User registration window reference
        self.registration_window = None

    def start(self):
        """Start the UI in a separate thread"""
        self.running = True
        self.ui_thread = threading.Thread(target=self._create_ui, daemon=True)
        self.ui_thread.start()

    def stop(self):
        """Stop the UI thread"""
        self.running = False
        if self.root:
            self.root.quit()

    def _create_ui(self):
        """Create and run the Tkinter UI"""
        self.root = tk.Tk()
        self.root.title("SecureFace Control Panel")
        self.root.geometry("400x600")  # Adjusted height
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Initialize UI variables after root is created
        # Detect available cameras
        available_cameras = detect_cameras()
        if not available_cameras:
            available_cameras = ["0"]  # Fallback to default if no cameras detected

        self.camera_source = tk.StringVar(
            value=str(available_cameras[0]) if available_cameras else "0"
        )
        self.width = tk.StringVar(value="640")
        self.height = tk.StringVar(value="480")
        self.fps = tk.StringVar(value="60")
        self.processing_enabled = tk.BooleanVar(
            value=True
        )  # Keep processing enabled but paused

        # Face detection parameters
        self.face_margin_ratio = tk.StringVar(value="0.8")
        self.face_rect_thickness = tk.StringVar(value="2")
        self.landmark_radius = tk.StringVar(value="2")

        # Preprocessing parameters
        self.convert_to_rgb = tk.BooleanVar(value=True)
        self.processing_mode = tk.StringVar(
            value="normal"
        )  # normal, aligned, fullframe
        self.target_width = tk.StringVar(value="112")
        self.target_height = tk.StringVar(value="112")

        # Recognition parameters
        self.continuous_scanning = tk.BooleanVar(value=False)
        self.recognition_threshold = tk.StringVar(value="1.0")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame, text="SecureFace Control Panel", font=("Arial", 12, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Camera settings section
        camera_label = ttk.Label(
            main_frame, text="Camera Settings", font=("Arial", 10, "bold")
        )
        camera_label.pack(anchor=tk.W, pady=(0, 5))

        camera_frame = ttk.LabelFrame(main_frame, text="Camera Configuration")
        camera_frame.pack(fill=tk.X, pady=(0, 10))

        # Camera source
        source_frame = ttk.Frame(camera_frame)
        source_frame.pack(fill=tk.X, pady=5)
        ttk.Label(source_frame, text="Camera Source:").pack(side=tk.LEFT)
        # Use detected cameras for the dropdown
        camera_sources = [str(i) for i in detect_cameras()] or ["0"]
        self.camera_combo = ttk.Combobox(
            source_frame,
            textvariable=self.camera_source,
            values=camera_sources,
            state="readonly",
            width=5,
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(5, 0))

        # Refresh cameras button
        refresh_btn = ttk.Button(
            source_frame, text="Refresh", command=self._refresh_cameras, width=8
        )
        refresh_btn.pack(side=tk.RIGHT)

        # Resolution
        resolution_frame = ttk.Frame(camera_frame)
        resolution_frame.pack(fill=tk.X, pady=5)
        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT)
        width_frame = ttk.Frame(resolution_frame)
        width_frame.pack(side=tk.RIGHT)
        ttk.Entry(width_frame, textvariable=self.width, width=6).pack(side=tk.LEFT)
        ttk.Label(width_frame, text="x").pack(side=tk.LEFT, padx=2)
        ttk.Entry(width_frame, textvariable=self.height, width=6).pack(side=tk.LEFT)

        # FPS
        fps_frame = ttk.Frame(camera_frame)
        fps_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        ttk.Entry(fps_frame, textvariable=self.fps, width=8).pack(side=tk.RIGHT)

        # Processing settings section
        processing_label = ttk.Label(
            main_frame, text="Processing Settings", font=("Arial", 10, "bold")
        )
        processing_label.pack(anchor=tk.W, pady=(0, 5))

        processing_frame = ttk.LabelFrame(main_frame, text="Processing Configuration")
        processing_frame.pack(fill=tk.X, pady=(0, 10))

        # Processing toggle
        ttk.Checkbutton(
            processing_frame, text="Enable Processing", variable=self.processing_enabled
        ).pack(anchor=tk.W, pady=5)

        # Face detection settings section
        face_label = ttk.Label(
            main_frame, text="Face Detection Settings", font=("Arial", 10, "bold")
        )
        face_label.pack(anchor=tk.W, pady=(0, 5))

        face_frame = ttk.LabelFrame(main_frame, text="Face Detection Configuration")
        face_frame.pack(fill=tk.X, pady=(0, 10))

        # Face margin ratio
        margin_frame = ttk.Frame(face_frame)
        margin_frame.pack(fill=tk.X, pady=5)
        ttk.Label(margin_frame, text="Face Margin Ratio:").pack(side=tk.LEFT)
        ttk.Entry(margin_frame, textvariable=self.face_margin_ratio, width=8).pack(
            side=tk.RIGHT
        )

        # Rectangle thickness
        rect_frame = ttk.Frame(face_frame)
        rect_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rect_frame, text="Rectangle Thickness:").pack(side=tk.LEFT)
        ttk.Entry(rect_frame, textvariable=self.face_rect_thickness, width=8).pack(
            side=tk.RIGHT
        )

        # Landmark radius
        landmark_frame = ttk.Frame(face_frame)
        landmark_frame.pack(fill=tk.X, pady=5)
        ttk.Label(landmark_frame, text="Landmark Radius:").pack(side=tk.LEFT)
        ttk.Entry(landmark_frame, textvariable=self.landmark_radius, width=8).pack(
            side=tk.RIGHT
        )

        # Preprocessing settings section
        preprocessing_label = ttk.Label(
            main_frame, text="Preprocessing Settings", font=("Arial", 10, "bold")
        )
        preprocessing_label.pack(anchor=tk.W, pady=(0, 5))

        preprocessing_frame = ttk.LabelFrame(
            main_frame, text="Preprocessing Configuration"
        )
        preprocessing_frame.pack(fill=tk.X, pady=(0, 10))

        # Mode selection
        mode_frame = ttk.Frame(preprocessing_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(mode_frame, text="Processing Mode:").pack(anchor=tk.W)

        # Radio buttons for mode selection
        self.processing_mode = tk.StringVar(
            value="normal"
        )  # normal, aligned, fullframe
        ttk.Radiobutton(
            mode_frame,
            text="Normal (Detect+Crop+Recognize)",
            variable=self.processing_mode,
            value="normal",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            mode_frame,
            text="Aligned Face (Skip Detection)",
            variable=self.processing_mode,
            value="aligned",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            mode_frame,
            text="Full Frame (Experimental)",
            variable=self.processing_mode,
            value="fullframe",
        ).pack(anchor=tk.W)

        # Detector selection
        detector_frame = ttk.Frame(preprocessing_frame)
        detector_frame.pack(fill=tk.X, pady=5)
        ttk.Label(detector_frame, text="Face Detector:").pack(anchor=tk.W)

        # Radio buttons for detector selection
        self.detector_type = tk.StringVar(
            value="retinaface"
        )  # retinaface, insightface, or scrfd
        ttk.Radiobutton(
            detector_frame,
            text="RetinaFace",
            variable=self.detector_type,
            value="retinaface",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            detector_frame,
            text="InsightFace",
            variable=self.detector_type,
            value="insightface",
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            detector_frame,
            text="SCRFD",
            variable=self.detector_type,
            value="scrfd",
        ).pack(anchor=tk.W)

        # RGB conversion toggle
        ttk.Checkbutton(
            preprocessing_frame, text="Convert to RGB", variable=self.convert_to_rgb
        ).pack(anchor=tk.W, pady=5)

        # Target size
        size_frame = ttk.Frame(preprocessing_frame)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Target Size:").pack(side=tk.LEFT)
        width_frame = ttk.Frame(size_frame)
        width_frame.pack(side=tk.RIGHT)
        ttk.Entry(width_frame, textvariable=self.target_width, width=6).pack(
            side=tk.LEFT
        )
        ttk.Label(width_frame, text="x").pack(side=tk.LEFT, padx=2)
        ttk.Entry(width_frame, textvariable=self.target_height, width=6).pack(
            side=tk.LEFT
        )

        # Recognition settings section
        recognition_label = ttk.Label(
            main_frame, text="Recognition Settings", font=("Arial", 10, "bold")
        )
        recognition_label.pack(anchor=tk.W, pady=(0, 5))

        recognition_frame = ttk.LabelFrame(
            main_frame, text="Face Recognition Configuration"
        )
        recognition_frame.pack(fill=tk.X, pady=(0, 10))

        # Continuous scanning toggle
        ttk.Checkbutton(
            recognition_frame,
            text="Enable Continuous Scanning",
            variable=self.continuous_scanning,
        ).pack(anchor=tk.W, pady=5)

        # Recognition threshold
        threshold_frame = ttk.Frame(recognition_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        ttk.Label(threshold_frame, text="Recognition Threshold:").pack(side=tk.LEFT)
        ttk.Entry(
            threshold_frame, textvariable=self.recognition_threshold, width=8
        ).pack(side=tk.RIGHT)

        # Control buttons section
        control_label = ttk.Label(
            main_frame, text="Controls", font=("Arial", 10, "bold")
        )
        control_label.pack(anchor=tk.W, pady=(0, 5))

        control_frame = ttk.LabelFrame(main_frame, text="Control Buttons")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Camera stream control buttons
        camera_control_frame = ttk.Frame(control_frame)
        camera_control_frame.pack(fill=tk.X, pady=5)
        ttk.Label(camera_control_frame, text="Camera Stream:").pack(side=tk.LEFT)
        self.camera_btn = ttk.Button(
            camera_control_frame, text="Pause", command=self._toggle_camera
        )
        self.camera_btn.pack(side=tk.RIGHT)

        # Processing control buttons
        processing_control_frame = ttk.Frame(control_frame)
        processing_control_frame.pack(fill=tk.X, pady=5)
        ttk.Label(processing_control_frame, text="Processing:").pack(side=tk.LEFT)
        self.processing_btn = ttk.Button(
            processing_control_frame, text="Pause", command=self._toggle_processing
        )
        self.processing_btn.pack(side=tk.RIGHT)

        # Apply button
        apply_btn = ttk.Button(
            main_frame, text="Apply Settings", command=self._apply_settings
        )
        apply_btn.pack(side=tk.RIGHT, pady=(0, 5))

        # Register new user button
        register_btn = ttk.Button(
            main_frame, text="Register New User", command=self._open_add_user_form
        )
        register_btn.pack(side=tk.RIGHT, padx=(0, 10), pady=(0, 5))

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, pady=(0, 5))

        # Start the UI loop
        self.root.mainloop()

    def _toggle_camera(self):
        """Toggle camera streaming"""
        self.camera_streaming = not self.camera_streaming
        if self.camera_streaming:
            self.camera_btn.config(text="Pause")
            self.status_label.config(text="Camera streaming resumed")
        else:
            self.camera_btn.config(text="Start")
            self.status_label.config(text="Camera streaming paused")

    def _toggle_processing(self):
        """Toggle frame processing"""
        self.processing_active = not self.processing_active
        if self.processing_active:
            self.processing_btn.config(text="Pause")
            self.status_label.config(text="Processing resumed")
        else:
            self.processing_btn.config(text="Start")
            self.status_label.config(text="Processing paused")

    def _apply_settings(self):
        """Send configuration to the main application"""
        try:
            config = {
                "camera_source": int(self.camera_source.get()),
                "width": int(self.width.get()),
                "height": int(self.height.get()),
                "fps": int(self.fps.get()),
                "processing_enabled": self.processing_enabled.get(),
                "camera_streaming": self.camera_streaming,
                "processing_active": self.processing_active,
                "face_margin_ratio": float(self.face_margin_ratio.get()),
                "face_rect_thickness": int(self.face_rect_thickness.get()),
                "landmark_radius": int(self.landmark_radius.get()),
                "processing_mode": self.processing_mode.get(),
                "detector_type": self.detector_type.get(),
                "convert_to_rgb": self.convert_to_rgb.get(),
                "target_width": int(self.target_width.get()),
                "target_height": int(self.target_height.get()),
                "continuous_scanning": self.continuous_scanning.get(),
                "recognition_threshold": float(self.recognition_threshold.get()),
            }

            # Put config in queue for main thread to process
            self.config_queue.put(config)
            self.status_label.config(text="Settings applied")
        except ValueError as e:
            self.status_label.config(text=f"Invalid input values: {str(e)}")
        except Exception as e:
            self.status_label.config(text=f"Error applying settings: {str(e)}")

    def _on_closing(self):
        """Handle window closing"""
        self.running = False
        self.root.destroy()

    def update_status(self, text):
        """Update status label from main thread"""
        if self.root and self.running:
            self.root.after(0, lambda: self.status_label.config(text=text))

    def set_captured_frame(self, frame):
        """Set the captured frame for user registration"""
        print(f"📥 UI Controller received captured frame - shape: {frame.shape}")
        # Pass the captured frame to the registration window if it exists
        if self.registration_window:
            print("📤 Forwarding frame to registration window")
            try:
                self.registration_window.set_captured_frame(frame)
                print("✅ Frame successfully forwarded to registration window")
            except Exception as e:
                error_msg = f"Failed to set captured frame: {str(e)}"
                print(f"❌ {error_msg}")
                import tkinter.messagebox as messagebox

                messagebox.showerror("Error", error_msg)
        else:
            print("⚠️ No registration window found to receive the frame")

    def _open_add_user_form(self):
        """Open the user registration form"""
        # Import here to avoid circular imports
        from user_registration import UserRegistrationWindow

        # Create and show the registration window
        try:
            self.registration_window = UserRegistrationWindow(
                self.root, self.config_queue
            )
        except Exception as e:
            import tkinter.messagebox as messagebox

            messagebox.showerror("Error", f"Failed to open registration form: {str(e)}")

    def _refresh_cameras(self):
        """Refresh the list of available cameras"""
        # Detect available cameras
        available_cameras = [str(i) for i in detect_cameras()] or ["0"]

        # Update the combobox values
        self.camera_combo["values"] = available_cameras

        # If current selection is not in the new list, select the first one
        if self.camera_source.get() not in available_cameras:
            self.camera_source.set(available_cameras[0])

        self.status_label.config(
            text=f"Cameras refreshed. Available: {', '.join(available_cameras)}"
        )
