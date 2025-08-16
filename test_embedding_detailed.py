#!/usr/bin/env python3

import cv2
import os
from embedder import embedder
import numpy as np

def test_embedding_from_file(image_path):
    \"\"\"Test embedding generation from a saved user image\"\"\"
    print(f\"\\n🔍 Testing embedding generation for: {image_path}\")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f\"❌ File not found: {image_path}\")
        return None
        
    # Load the image
    print(\"📥 Loading image...\")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f\"❌ Failed to load image: {image_path}\")
        return None
        
    print(f\"✅ Image loaded successfully - shape: {frame.shape}\")
    
    # Test the embedding generation
    print(\"🧠 Generating embedding...\")
    try:
        # Convert BGR to RGB (InsightFace expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f\"🔄 Color conversion completed - shape: {img_rgb.shape}\")
        
        # Get embedding using the global embedder
        print(\"🔍 Detecting faces and generating embedding using InsightFace...\")
        faces = embedder.app.get(img_rgb)
        print(f\"👤 Face detection completed - found {len(faces)} face(s)\")
        
        if len(faces) > 0:
            embedding = faces[0].embedding
            print(f\"✅ Embedding generated successfully - shape: {embedding.shape}\")
            print(f\"📏 Embedding norm: {np.linalg.norm(embedding)}\")
            
            # Show face detection info
            face = faces[0]
            print(f\"📊 Face confidence: {face.det_score}\")
            print(f\"📍 Face bounding box: {face.bbox}\")
            if hasattr(face, 'landmark_2d_106'):
                print(f\" facial landmarks detected\")
            
            return embedding
        else:
            print(\"⚠️ No faces detected in the image\")
            # Save debug image to see what went wrong
            debug_dir = \"debug_images\"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            debug_filename = f\"{debug_dir}/debug_no_face_user_image.jpg\"
            cv2.imwrite(debug_filename, frame)
            print(f\"💾 Saved debug image as {debug_filename} for inspection\")
            return None
            
    except Exception as e:
        print(f\"❌ Error generating embedding: {e}\")
        import traceback
        traceback.print_exc()
        return None

def list_user_images():
    \"\"\"List available user images\"\"\"
    user_images_dir = \"user_images\"
    if not os.path.exists(user_images_dir):
        print(f\"❌ Directory not found: {user_images_dir}\")
        return []
        
    images = [f for f in os.listdir(user_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f\"⚠️ No images found in {user_images_dir}\")
        return []
        
    print(f\"📁 Found {len(images)} image(s) in {user_images_dir}:\")
    for i, img in enumerate(images, 1):
        print(f\"  {i}. {img}\")
    return images

def test_all_user_images():
    \"\"\"Test embedding generation for all user images\"\"\"
    print(\"🧪 User Image Embedding Test - All Images\")
    print(\"=\" * 50)
    
    # List available images
    images = list_user_images()
    
    if not images:
        print(\"🚫 No images to test\")
        return
        
    # Test all images
    results = []
    for img_name in images:
        image_path = os.path.join(\"user_images\", img_name)
        embedding = test_embedding_from_file(image_path)
        results.append((img_name, embedding is not None))
        
    # Summary
    print(f\"\\n📊 Summary:\")
    print(f\"{'Image':<25} {'Status':<10}\")
    print(\"-\" * 35)
    success_count = 0
    for img_name, success in results:
        status = \"✅ Success\" if success else \"❌ Failed\"
        print(f\"{img_name:<25} {status:<10}\")
        if success:
            success_count += 1
            
    print(f\"\\n📈 {success_count}/{len(images)} images processed successfully\")

if __name__ == \"__main__\"
:
    test_all_user_images()