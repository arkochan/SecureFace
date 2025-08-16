#!/usr/bin/env python3
"""
Test script to verify database setup
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.init_db import create_tables
from database.db import SecureFaceDB

def test_database():
    """Test the database setup"""
    print("Creating database tables...")
    create_tables()
    
    print("Testing database operations...")
    with SecureFaceDB() as db:
        # Test creating a role
        print("Creating a test role...")
        role_id = db.create_role("Test Role", 25, "A test role for verification")
        print(f"Created role with ID: {role_id}")
        
        # Test creating a user
        print("Creating a test user...")
        user_id = db.create_user("Test User", role_id, "IT", "/path/to/image.jpg")
        print(f"Created user with ID: {user_id}")
        
        # Test retrieving the user
        print("Retrieving user...")
        user = db.get_user_by_id(user_id)
        print(f"Retrieved user: {user}")
        
        # Test updating the user
        print("Updating user...")
        db.update_user(user_id, full_name="Updated Test User")
        
        # Test retrieving the updated user
        print("Retrieving updated user...")
        updated_user = db.get_user_by_id(user_id)
        print(f"Updated user: {updated_user}")
        
        # Test creating a recognition log
        print("Creating a recognition log...")
        log_id = db.create_recognition_log(user_id, 1, "ALLOWED", 0.95)
        print(f"Created log with ID: {log_id}")
        
        # Test retrieving logs
        print("Retrieving logs...")
        logs = db.get_logs_by_user(user_id)
        print(f"Retrieved {len(logs)} log(s)")
        
        # Test getting user with role
        print("Getting user with role info...")
        user_with_role = db.get_user_with_role(user_id)
        print(f"User with role: {user_with_role}")
        
        print("All tests passed!")

if __name__ == "__main__":
    test_database()