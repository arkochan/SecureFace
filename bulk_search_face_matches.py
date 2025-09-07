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
import glob


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
            print(
                f"📏 Normalized embedding norm: {np.linalg.norm(normalized_embedding)}"
            )
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
        _, buffer = cv2.imencode(".jpg", resized_img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return img_base64
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None


def generate_html_report(
    query_image_path, results, output_path="face_search_results.html"
):
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
        "    <h2>Query Image</h2>",
    ]

    if query_image_base64:
        html_lines.append(
            f'    <img src="data:image/jpeg;base64,{query_image_base64}" alt="Query Image" style="max-width: 300px;">'
        )
    else:
        html_lines.append("    <p>Error loading query image</p>")

    html_lines.append("    ")
    html_lines.append("    <h2>Search Results</h2>")

    if not results:
        html_lines.append("    <p>No matching faces found in the database</p>")
    else:
        html_lines.append("    <table border='1' cellpadding='5' cellspacing='0'>")
        html_lines.append(
            "        <tr><th>Rank</th><th>Match Image</th><th>User ID</th><th>User Name</th><th>Department</th><th>Distance</th><th>Registered At</th></tr>"
        )

        for i, (faiss_id, distance, user_id, created_at) in enumerate(results, 1):
            # Get user information
            user = get_user_info(user_id)
            user_name = user["full_name"] if user and user["full_name"] else "N/A"
            department = user["department"] if user and user["department"] else "N/A"
            image_path = user["image_path"] if user and user["image_path"] else None
            print(image_path)

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
            html_lines.append(
                f"            <td>{created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'}</td>"
            )
            html_lines.append("        </tr>")

        html_lines.append("    </table>")

    html_lines.append("</body>")
    html_lines.append("</html>")

    # Join all HTML lines
    html_content = "\r".join(html_lines)

    # Write HTML to file
    try:
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"✅ HTML report generated: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing HTML report: {e}")
        return False


def process_single_image(image_path, output_path, top_k):
    """Process a single image and generate its report"""
    print(f"🔍 SecureFace Face Search Utility")
    print("=" * 40)

    # Validate input image path
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False

    # Load and embed the image
    embedding = load_and_embed_image(image_path)
    if embedding is None:
        print(f"❌ Failed to generate embedding for the image: {image_path}")
        return False

    # Search for similar faces
    results = search_similar_faces(embedding, k=top_k)

    # Generate HTML report
    success = generate_html_report(image_path, results, output_path)

    if success:
        print(f"\r✅ Face search completed! Results saved to {output_path}")
        return True
    else:
        print(f"\r❌ Failed to generate HTML report for {image_path}")
        return False


def process_directory(input_dir, output_dir, max_count, top_k, file_pattern="*.jpg"):
    """Process multiple images from a directory"""
    print(f"🔍 Processing images from directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Max count: {max_count}")
    print(f"📊 Top-k matches: {top_k}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    search_pattern = os.path.join(input_dir, file_pattern)
    image_files = glob.glob(search_pattern)

    # Sort files for consistent ordering
    image_files.sort()

    # Limit to max_count if specified
    if max_count > 0:
        image_files = image_files[:max_count]

    print(f"📷 Found {len(image_files)} images to process")

    # Process each image
    success_count = 0
    for i, image_path in enumerate(image_files):
        print(f"\r[{i + 1}/{len(image_files)}] Processing {image_path}")

        # Generate output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.html")

        # Process the image
        if process_single_image(image_path, output_path, top_k):
            success_count += 1

    print(
        f"\r🏁 Processing complete! Successfully processed {success_count}/{len(image_files)} images"
    )
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Search for similar faces in the SecureFace database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python bulk_search_face_matches.py image.jpg                    # Search with image.jpg, output to face_search_results.html
  python bulk_search_face_matches.py image.jpg -o results.html    # Search with custom output file
  python bulk_search_face_matches.py image.jpg -k 10              # Search for top 10 matches
  python bulk_search_face_matches.py -d demo_entity/              # Process all images in directory
  python bulk_search_face_matches.py -d demo_entity/ -m 20        # Process first 20 images in directory
  python bulk_search_face_matches.py -d demo_entity/ -o results/  # Save results to results directory""",
    )

    # Original single image arguments
    parser.add_argument(
        "image_path", nargs="?", help="Path to the image file to search for"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="face_search_results.html",
        help="Output HTML file path (default: face_search_results.html)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)",
    )

    # New bulk processing arguments
    parser.add_argument(
        "-d", "--directory", help="Directory containing images to process"
    )
    parser.add_argument(
        "-m",
        "--max-count",
        type=int,
        default=0,
        help="Maximum number of images to process (0 for all)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for HTML reports (default: results)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="*.jpg",
        help="File pattern to match (default: *.jpg)",
    )

    args = parser.parse_args()

    # Check if we're processing a directory or single image
    if args.directory:
        # Process directory
        success_count = process_directory(
            args.directory, args.output_dir, args.max_count, args.top_k, args.pattern
        )

        if success_count == 0:
            print("❌ No images were successfully processed")
            sys.exit(1)
    else:
        # Process single image (original behavior)
        if not args.image_path:
            parser.print_help()
            sys.exit(1)

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
            print(f"\r✅ Face search completed! Results saved to {args.output}")
        else:
            print(f"\r❌ Failed to generate HTML report")
            sys.exit(1)


if __name__ == "__main__":
    main()
