#!/usr/bin/env python3

import cv2
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import base64
from embedder import FaceEmbedder
import vector_db
from database.db import SecureFaceDB

def normalize_embedding(embedding):
    """Normalize embedding to unit length"""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def load_and_embed_image(image_path):
    """Load an image and generate its embedding"""
    print(f"🔍 Loading and embedding image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
        
    # Load the image
    print("📥 Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
        
    print(f"✅ Image loaded successfully - shape: {frame.shape}")
    
    # Initialize the embedder
    embedder = FaceEmbedder()
    
    # Generate embedding
    print("🧠 Generating embedding...")
    try:
        # Convert BGR to RGB (InsightFace expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"🔄 Color conversion completed - shape: {img_rgb.shape}")
        
        # Get embedding using the embedder
        print("🔍 Detecting faces and generating embedding using InsightFace...")
        faces = embedder.app.get(img_rgb)
        print(f"👤 Face detection completed - found {len(faces)} face(s)")
        
        if len(faces) > 0:
            embedding = faces[0].embedding
            print(f"✅ Embedding generated successfully - shape: {embedding.shape}")
            # Normalize the embedding for better similarity matching
            normalized_embedding = normalize_embedding(embedding)
            print(f"📏 Normalized embedding norm: {np.linalg.norm(normalized_embedding)}")
            return normalized_embedding
        else:
            print("⚠️ No faces detected in the image")
            return None
            
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up embedder
        embedder.stop()

def search_similar_faces(embedding, k=5):
    """Search for similar faces in the vector database"""
    print(f"🔍 Searching for similar faces...")
    
    # Initialize FAISS index
    vector_db.init_index(dim=512, index_path="faiss_index.bin")
    
    # Normalize the query embedding
    normalized_embedding = normalize_embedding(embedding)
    print(f"📏 Query embedding norm: {np.linalg.norm(normalized_embedding)}")
    
    # Search for similar embeddings
    results = vector_db.search_embeddings(normalized_embedding, k=k)
    
    if results:
        print(f"✅ Found {len(results)} similar faces")
        return results
    else:
        print("⚠️ No similar faces found")
        return []

def get_user_info(user_id):
    """Get user information from the database"""
    try:
        with SecureFaceDB() as db:
            user = db.get_user_by_id(user_id)
            return user
    except Exception as e:
        print(f"❌ Error fetching user info for ID {user_id}: {e}")
        return None

def encode_image_to_base64(image_path):
    """Encode an image to base64 for embedding in HTML"""
    try:
        if not os.path.exists(image_path):
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize image to a standard height for display
        height = 100
        aspect_ratio = img.shape[1] / img.shape[0]
        width = int(height * aspect_ratio)
        resized_img = cv2.resize(img, (width, height))
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', resized_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def generate_html_report(query_image_path, results, output_path="face_search_results.html"):
    """Generate an HTML report with the search results"""
    print(f"📊 Generating HTML report: {output_path}")
    
    # Read the query image and encode it
    query_image_base64 = encode_image_to_base64(query_image_path)
    
    # Format the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>SecureFace - Face Search Results</title>",
        '    <meta charset="UTF-8">',
        "</head>",
        "<body>",
        "    <h1>SecureFace Face Search Results</h1>",
        f"    <p>Generated at: {current_time}</p>",
        "    ",
        "    <h2>Query Image</h2>"
    ]
    
    if query_image_base64:
        html_lines.append(f'    <img src="data:image/jpeg;base64,{query_image_base64}" alt="Query Image" style="max-width: 300px;">')
    else:
        html_lines.append("    <p>Error loading query image</p>")
    
    html_lines.append("    ")
    html_lines.append("    <h2>Search Results</h2>")
    
    if not results:
        html_lines.append("    <p>No matching faces found in the database</p>")
    else:
        html_lines.append("    <table border='1' cellpadding='5' cellspacing='0'>")
        html_lines.append("        <tr><th>Rank</th><th>Match Image</th><th>User ID</th><th>User Name</th><th>Department</th><th>Distance</th><th>Registered At</th></tr>")
        
        for i, (faiss_id, distance, user_id, created_at) in enumerate(results, 1):
            # Get user information
            user = get_user_info(user_id)
            user_name = user['full_name'] if user and user['full_name'] else "N/A"
            department = user['department'] if user and user['department'] else "N/A"
            image_path = user['image_path'] if user and user['image_path'] else None
            
            # Encode user image if available
            user_image_html = "No Image"
            if image_path and os.path.exists(image_path):
                user_image_base64 = encode_image_to_base64(image_path)
                if user_image_base64:
                    user_image_html = f'<img src="data:image/jpeg;base64,{user_image_base64}" alt="User Image" style="max-height: 100px;">'
            
            html_lines.append("        <tr>")
            html_lines.append(f"            <td>{i}</td>")
            html_lines.append(f"            <td>{user_image_html}</td>")
            html_lines.append(f"            <td>{user_id}</td>")
            html_lines.append(f"            <td>{user_name}</td>")
            html_lines.append(f"            <td>{department}</td>")
            html_lines.append(f"            <td>{distance:.4f}</td>")
            html_lines.append(f"            <td>{created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'}</td>")
            html_lines.append("        </tr>")
        
        html_lines.append("    </table>")
    
    html_lines.append("</body>")
    html_lines.append("</html>")
    
    # Join all HTML lines
    html_content = "\n".join(html_lines)
    
    # Write HTML to file
    try:
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"✅ HTML report generated: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing HTML report: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Search for similar faces in the SecureFace database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_face_matches.py image.jpg                    # Search with image.jpg, output to face_search_results.html
  python search_face_matches.py image.jpg -o results.html    # Search with custom output file
  python search_face_matches.py image.jpg -k 10              # Search for top 10 matches
        """
    )
    
    parser.add_argument("image_path", help="Path to the image file to search for")
    parser.add_argument("-o", "--output", default="face_search_results.html", 
                        help="Output HTML file path (default: face_search_results.html)")
    parser.add_argument("-k", "--top-k", type=int, default=5, 
                        help="Number of top matches to return (default: 5)")
    
    args = parser.parse_args()
    
    print("🔍 SecureFace Face Search Utility")
    print("=" * 40)
    
    # Validate input image path
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Load and embed the image
    embedding = load_and_embed_image(args.image_path)
    if embedding is None:
        print("❌ Failed to generate embedding for the image")
        sys.exit(1)
    
    # Search for similar faces
    results = search_similar_faces(embedding, k=args.top_k)
    
    # Generate HTML report
    success = generate_html_report(args.image_path, results, args.output)
    
    if success:
        print(f"\n✅ Face search completed! Results saved to {args.output}")
    else:
        print(f"\n❌ Failed to generate HTML report")
        sys.exit(1)

if __name__ == "__main__":
    main()