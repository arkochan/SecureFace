import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from glob import glob

print("✅ InsightFace imported")
print("✅ FaceAnalysis imported")

# Initialize with CPU only
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
print("🔧 FaceAnalysis instance created")

# Prepare
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ App prepared successfully")
# Supported image extensions
extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]

# Get all image paths in current directory
undetected_list = []
image_paths = []
for ext in extensions:
    image_paths.extend(glob(ext))

if not image_paths:
    print("No images found in current directory.")

for image_path in image_paths:
    # Load a REAL cropped face image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face analysis
    faces = app.get(img_rgb)
    if len(faces) > 1:
        print(f"🔍 Found {len(faces)} face(s)")

    if len(faces) == 0:
        undetected_list.append(image_path)
    else:
        embedding = faces[0].embedding
        print("✅ Embedding done:", image_path)
        print("First 10 values:", embedding[:10])
        # np.save("face_embedding.npy", embedding)
        # print("💾 Embedding saved to 'face_embedding.npy'")

print("❌ could not detect for", undetected_list)
