# SecureFace

A real-time face detection and recognition system using RetinaFace for detection and ArcFace for recognition.

## Setup Instructions

### Prerequisites
- Anaconda or Miniconda installed
- Python 3.10

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arkochan/secureface
   cd secureface
   ```

2. Create the conda environment using the environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate SecureFace
   ```

### Running the Application

Once the environment is set up and activated, run the main application:
```bash
python main.py
```

### Controls

The application provides a GUI control panel where you can adjust various parameters:
- Camera settings (source, resolution, FPS)
- Face detection parameters (margin ratio, rectangle thickness, landmark radius)
- Preprocessing settings (RGB conversion, target size)
- Processing modes (Normal, Aligned Face, Full Frame)

### Processing Modes

1. **Normal Mode**: Detects faces in the video stream, crops them, and generates embeddings
2. **Aligned Face Mode**: Skips face detection and assumes input images are already aligned faces
3. **Full Frame Mode**: Sends entire frames to the embedding processor (experimental)

### Debugging

Debug images are saved in the `debug_images` directory when face detection or embedding fails.

## Project Structure

- `main.py`: Main application entry point
- `frame_processor.py`: Handles video frame processing, face detection, and cropping
- `embedder.py`: Manages face embedding generation using ArcFace
- `ui_controller.py`: Controls the GUI interface
- `stream.py`: Handles video streaming from camera
- `test-arc-cpu.py`: Test script for ArcFace embedding
- `environment.yml`: Conda environment specification
- `debug_images/`: Directory for debug image storage

## Dependencies

- insightface: For ArcFace face recognition
- OpenCV: For image processing and camera access
- NumPy: For numerical operations
- Tkinter: For GUI interface
- retinaface: For face detection
