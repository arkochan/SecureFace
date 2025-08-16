import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from glob import glob
import argparse  # To handle command line arguments
import sys

# --- Import and initialize vector database ---
import vector_db

# Initialize FAISS index (load from disk if exists, otherwise create new)
# You might want to specify a path to save/load the index
vector_db.init_index(dim=512, index_path="faiss_index.bin")
print("🔧 FAISS Vector Database initialized")
# ---------------------------------------------

print("✅ InsightFace imported")
print("✅ FaceAnalysis imported")

# --- Handle command line arguments ---
parser = argparse.ArgumentParser(
    description="Process face embeddings and store in FAISS DB."
)
parser.add_argument(
    "--register", type=str, help="Path to the image to register as the template."
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.8,
    help="Distance threshold for matching (default: 1.0). Lower is stricter.",
)
args = parser.parse_args()
# ---------------------------------------

# Initialize with CPU only
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
print("🔧 FaceAnalysis instance created")

# Prepare
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ App prepared successfully")

# --- Register Mode ---
if args.register:
    register_path = args.register
    if not os.path.exists(register_path):
        print(f"❌ Registration image not found: {register_path}")
        sys.exit(1)

    # Load the registration image
    img = cv2.imread(register_path)
    if img is None:
        print(f"❌ Could not read registration image: {register_path}")
        sys.exit(1)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face analysis
    faces = app.get(img_rgb)
    if len(faces) == 0:
        print(f"❌ No faces detected in registration image: {register_path}")
        sys.exit(1)
    elif len(faces) > 1:
        print(
            f"⚠️ Multiple faces found in registration image {register_path}. Using the first one."
        )

    embedding = faces[0].embedding
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    print("✅ Embedding generated for registration image:", register_path)
    print("📏 Embedding shape:", embedding.shape)

    # --- Store the registration embedding in FAISS vector database ---
    # Use a special user_id (e.g., -1) to denote the registered template
    template_user_id = -1
    faiss_id = vector_db.add_embedding(embedding, template_user_id)
    if faiss_id != -1:
        print(
            f"💾 Registration embedding stored in FAISS DB with internal ID: {faiss_id} and user_id: {template_user_id}"
        )
        # Save the index after adding the template
        vector_db.save_index("faiss_index.bin")
        print("💾 FAISS index saved to 'faiss_index.bin'")
    else:
        print(f"❌ Failed to store registration embedding for {register_path}")
        sys.exit(1)

    print(
        "✅ Registration completed. You can now run the script without --register to compare other images."
    )
    sys.exit(0)  # Exit after registration


def generate_html_report(matched_images, unmatched_images, threshold):
    """Generate an HTML report showing matched and unmatched images."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Face Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1, h2 {{
            color: #333;
        }}
        .image-container {{
            display: inline-block;
            margin: 10px;
            text-align: center;
        }}
        .image-container img {{
            max-width: 200px;
            max-height: 200px;
        }}
        .match {{
            border: 3px solid green;
        }}
        .no-match {{
            border: 3px solid red;
        }}
        .distance {{
            font-weight: bold;
        }}
        .threshold {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Face Comparison Report</h1>
    
    <div class="threshold">
        <p><strong>Threshold:</strong> {threshold}</p>
    </div>
    
    <h2>Matched Images ({len(matched_images)})</h2>
"""

    # Add matched images
    if matched_images:
        for image_path, distance in matched_images:
            html_content += f"""
    <div class="image-container match">
        <img src="{image_path}" alt="{image_path}">
        <p>{image_path}<br><span class="distance">Distance: {distance:.4f}</span></p>
    </div>
"""
    else:
        html_content += "    <p>No matched images found.</p>\n"

    html_content += f"""
    <h2>Unmatched Images ({len(unmatched_images)})</h2>
"""

    # Add unmatched images
    if unmatched_images:
        for image_path, distance in unmatched_images:
            html_content += f"""
    <div class="image-container no-match">
        <img src="{image_path}" alt="{image_path}">
        <p>{image_path}<br><span class="distance">Distance: {distance:.4f}</span></p>
    </div>
"""
    else:
        html_content += "    <p>No unmatched images found.</p>\n"

    html_content += """
</body>
</html>"""

    # Write HTML to file
    with open("face_comparison_report.html", "w") as f:
        f.write(html_content)


# --- Comparison Mode (if --register is not provided) ---
print("🔍 Running in comparison mode. Looking for a registered template...")

# --- Retrieve the registered template embedding ---
template_embedding = vector_db.get_template_embedding(template_user_id=-1)

if template_embedding is None:
    print(
        "❌ No registered template found. Please register a template image first using --register."
    )
    sys.exit(1)

print("✅ Retrieved registered template embedding.")

# --- Process images in the test_match directory for comparison ---
test_match_dir = "user_images"
if not os.path.exists(test_match_dir):
    print(f"⚠️ Test match directory '{test_match_dir}' not found.")
    sys.exit(0)  # Not an error, just no directory to compare

# Supported image extensions
extensions = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]

# Get all image paths in test_match directory
image_paths = []
for ext in extensions:
    image_paths.extend(glob(os.path.join(test_match_dir, ext)))

if not image_paths:
    print("⚠️ No images found in current directory for comparison.")
    sys.exit(0)  # Not an error, just no images to compare

print(f"🔍 Comparing {len(image_paths)} images against the registered template...")

# --- Track matched and unmatched images ---
matched_images = []
unmatched_images = []

# --- Compare each image against the template ---
for image_path in image_paths:
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        continue

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face analysis
    faces = app.get(img_rgb)
    if len(faces) == 0:
        print(f"❌ No faces detected in {image_path}")
        continue
    elif len(faces) > 1:
        print(
            f"⚠️ Multiple faces found in {image_path}. Using the first one for comparison."
        )

    embedding = faces[0].embedding
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    print(f"✅ Embedding generated for {image_path}")

    # --- Calculate distance to the template embedding ---
    # Using L2 distance (Euclidean distance)
    distance = np.linalg.norm(template_embedding - embedding)
    print(f"📏 Distance between {image_path} and template: {distance:.4f}")

    # --- Determine match based on threshold ---
    if distance < args.threshold:
        print(
            f"✅ MATCH: {image_path} is likely the same person as the template (Distance: {distance:.4f} < Threshold: {args.threshold})"
        )
        matched_images.append((image_path, distance))
    else:
        print(
            f"❌ NO MATCH: {image_path} is NOT the same person as the template (Distance: {distance:.4f} >= Threshold: {args.threshold})"
        )
        unmatched_images.append((image_path, distance))

print("\n🏁 Comparison process finished.")

# --- Generate HTML report ---
print("\n📊 Generating HTML report...")
generate_html_report(matched_images, unmatched_images, args.threshold)
print("✅ HTML report generated: face_comparison_report.html")
