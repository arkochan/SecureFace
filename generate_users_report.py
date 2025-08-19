#!/usr/bin/env python3

import json
import os
from datetime import datetime
from database.db import SecureFaceDB
import vector_db
import base64
import cv2

# Initialize FAISS index to get proper stats
vector_db.init_index(dim=512, index_path="faiss_index.bin")
print("🔧 FAISS Vector Database initialized for reporting")

def generate_users_report():
    """Generate JSON and HTML reports showing all registered users"""
    
    # Get all users with their role information
    print("📥 Fetching user data from database...")
    try:
        with SecureFaceDB() as db:
            users = db.get_all_users()
            if not users:
                print("⚠️ No users found in database")
                users = []
            
            # Get role information for better display
            roles = {}
            role_data = db.get_all_roles()
            if role_data:
                for role in role_data:
                    roles[role['role_id']] = {
                        'name': role['role_name'],
                        'access_level': role['access_level'],
                        'description': role['description']
                    }
            
        print(f"✅ Retrieved {len(users)} users from database")
        
    except Exception as e:
        print(f"❌ Error fetching user data: {e}")
        return
    
    # Get FAISS index statistics
    print("📊 Getting FAISS index statistics...")
    try:
        faiss_stats = vector_db.get_index_stats()
        print(f"✅ FAISS index stats: {faiss_stats}")
    except Exception as e:
        print(f"⚠️ Error getting FAISS stats: {e}")
        faiss_stats = {"initialized": False}
    
    # Prepare report data
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "statistics": {
            "total_users": len(users),
            "faiss_status": "Active" if faiss_stats.get("initialized", False) else "Inactive",
            "total_roles": len(roles),
            "face_embeddings": faiss_stats.get("total_vectors", 0) if faiss_stats.get("initialized", False) else 0
        },
        "users": [],
        "roles": roles
    }
    
    # Add user data
    for user in users:
        user_data = {
            "user_id": user['user_id'],
            "full_name": user['full_name'],
            "role_id": user['role_id'],
            "department": user['department'],
            "image_path": user['image_path'],
            "created_at": user['created_at'].isoformat() if user['created_at'] else None,
            "updated_at": user['updated_at'].isoformat() if user['updated_at'] else None,
            "role_name": roles.get(user['role_id'], {}).get('name') if user['role_id'] else None
        }
        report_data["users"].append(user_data)
    
    # Save JSON report
    json_filename = "registered_users_report.json"
    try:
        with open(json_filename, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"✅ JSON report generated: {json_filename}")
    except Exception as e:
        print(f"❌ Error writing JSON report: {e}")
        return
    
    # Generate simple HTML report
    html_filename = "registered_users_report.html"
    try:
        html_content = generate_simple_html(report_data)
        with open(html_filename, "w") as f:
            f.write(html_content)
        print(f"✅ HTML report generated: {html_filename}")
    except Exception as e:
        print(f"❌ Error writing HTML report: {e}")
        return
    
    print(f"\n📈 Report Summary:")
    print(f"   Total Users: {report_data['statistics']['total_users']}")
    print(f"   Face Embeddings: {report_data['statistics']['face_embeddings']}")
    print(f"   FAISS Status: {report_data['statistics']['faiss_status']}")

def generate_simple_html(report_data):
    """Generate a simple HTML report"""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>SecureFace - Registered Users Report</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>SecureFace Registered Users Report</h1>
    <p>Generated at: {}</p>
    
    <h2>System Statistics</h2>
    <ul>
        <li>Total Users: {}</li>
        <li>FAISS Status: {}</li>
        <li>User Roles: {}</li>
        <li>Face Embeddings: {}</li>
    </ul>
    
    <h2>Registered Users ({})</h2>
""".format(
        report_data["generated_at"],
        report_data["statistics"]["total_users"],
        report_data["statistics"]["faiss_status"],
        report_data["statistics"]["total_roles"],
        report_data["statistics"]["face_embeddings"],
        report_data["statistics"]["total_users"]
    )
    
    if not report_data["users"]:
        html += "    <p>No users registered.</p>\n"
    else:
        html += "    <table border='1' cellpadding='5' cellspacing='0'>\n"
        html += "        <tr><th>Image</th><th>User ID</th><th>Name</th><th>Department</th><th>Role</th><th>Registered</th></tr>\n"
        
        for user in report_data["users"]:
            image_html = "No Image"
            if user['image_path'] and os.path.exists(user['image_path']):
                try:
                    img = cv2.imread(user['image_path'])
                    # Resize image to a standard height
                    height = 100
                    aspect_ratio = img.shape[1] / img.shape[0]
                    width = int(height * aspect_ratio)
                    resized_img = cv2.resize(img, (width, height))
                    
                    # Encode image to base64
                    _, buffer = cv2.imencode('.jpg', resized_img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" alt="User Image">'
                except Exception as e:
                    print(f"Error processing image {user['image_path']}: {e}")
                    image_html = "Error loading image"

            html += "        <tr>\n"
            html += f"            <td>{image_html}</td>\n"
            html += f"            <td>{user['user_id']}</td>\n"
            html += f"            <td>{user['full_name']}</td>\n"
            html += f"            <td>{user['department'] or 'N/A'}</td>\n"
            html += f"            <td>{user['role_name'] or 'N/A'}</td>\n"
            html += f"            <td>{user['created_at'][:19] if user['created_at'] else 'N/A'}</td>\n"
            html += "        </tr>\n"
        
        html += "    </table>\n"
    
    html += "</body>\n</html>"
    return html

if __name__ == "__main__":
    print("📋 SecureFace Registered Users Report Generator")
    print("=" * 50)
    generate_users_report()
    print("\n✅ Report generation completed!")