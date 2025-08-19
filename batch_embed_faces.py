

import cv2
import os
import numpy as np
from embedder import FaceEmbedder
import vector_db
import time
from faker import Faker
from database.db import SecureFaceDB
import random

def embed_and_store_faces(limit=2000):
    """
    Scans the 'demo_faces' directory, generates embeddings for each face,
    creates a new user with fake data, and stores the embedding in the vector database.
    Stops after successfully processing the specified limit of images.
    """
    print("🚀 Starting batch embedding and user creation process...")
    
    # Initialize the embedder and vector database
    embedder = FaceEmbedder()
    vector_db.init_index(dim=512, index_path="faiss_index.bin")
    
    # Initialize Faker
    fake = Faker()
    
    faces_dir = "demo_faces"
    if not os.path.exists(faces_dir):
        print(f"❌ Directory not found: {faces_dir}")
        return

    image_files = []
    for root, _, files in os.walk(faces_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"⚠️ No images found in {faces_dir}")
        return

    print(f"🖼️ Found {len(image_files)} images to process.")
    
    success_count = 0
    start_time = time.time()

    with SecureFaceDB() as db:
        roles = db.get_all_roles()
        if not roles:
            print("❌ No roles found in the database. Please add roles first.")
            return

        for i, image_path in enumerate(image_files):
            if success_count >= limit:
                print(f"🏁 Reached embedding limit of {limit}. Stopping.")
                break

            print(f"--- Processing image {i+1}/{len(image_files)}: {image_path} ---")
            
            try:
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"⚠️ Could not read image: {image_path}")
                    continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get embedding using the direct insightface method
                faces = embedder.app.get(img_rgb)
                
                if len(faces) > 0:
                    embedding = faces[0].embedding
                    
                    # Normalize the embedding
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    
                    # Generate fake user data
                    full_name = fake.name()
                    departments = ['Engineering', 'HR', 'Sales', 'Marketing', 'Product']
                    department = random.choice(departments)
                    role_id = random.choice([role['role_id'] for role in roles])
                    
                    # Create the user
                    user_id = db.create_user(full_name, role_id, department, image_path)
                    
                    if user_id:
                        print(f"✅ User '{full_name}' created with user_id: {user_id}")
                        
                        # Add to vector database
                        faiss_id = vector_db.add_embedding(normalized_embedding, user_id)
                        if faiss_id != -1:
                            print(f"✅ Successfully embedded and stored user ID {user_id} with FAISS ID {faiss_id}")
                            success_count += 1
                        else:
                            print(f"❌ Failed to store embedding for user ID {user_id}")
                    else:
                        print("❌ Failed to create user in the database.")
                else:
                    print(f"⚠️ No face detected in {image_path}")

            except Exception as e:
                print(f"❌ An error occurred while processing {image_path}: {e}")

            if (i + 1) % 100 == 0:
                print(f"--- Progress: {success_count}/{limit} embeddings stored ---")

    end_time = time.time()
    print("\n" + "="*50)
    print("✅ Batch embedding and user creation process finished.")
    print(f"Successfully created {success_count} new users and embeddings.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    
    # Save the updated index
    print("💾 Saving FAISS index...")
    vector_db.save_index("faiss_index.bin")
    print("✅ Index saved.")

if __name__ == "__main__":
    embed_and_store_faces()

